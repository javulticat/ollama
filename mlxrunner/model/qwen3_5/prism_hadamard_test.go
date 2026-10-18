package qwen3_5

import (
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestParsePrismHadamardConfig(t *testing.T) {
	const header = `"schema_version": 2, "base_model_type": "qwen3_5",
		"tensor_namespace": "mlx-vlm-qwen3_5", "gdn_activation_layout": "grouped"`
	modules, err := parsePrismHadamardConfig([]byte(`{` + header + `, "modules": [
		{"path": "model.embed_tokens", "block": 512, "embedding": true, "dtype": "float16"},
		{"path": "model.layers.0.mlp.down_proj", "block": 0, "embedding": false, "dtype": "float16"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	want := []hadamardModule{
		{Path: "model.embed_tokens", Block: 512, Embedding: true, DType: "float16"},
		{Path: "model.layers.0.mlp.down_proj", DType: "float16"},
	}
	if len(modules) != len(want) || modules[0] != want[0] || modules[1] != want[1] {
		t.Fatalf("modules = %+v, want %+v", modules, want)
	}

	module := func(fields string) string {
		return `{"path": "model.layers.0.mlp.up_proj", "embedding": false, ` + fields + `}`
	}
	for _, tt := range []struct {
		name, config, wantErr string
	}{
		{"schema 1", `{"schema_version": 1, "base_model_type": "qwen3_5"}`, "unsupported Hadamard pack"},
		{"other base", `{"schema_version": 2, "base_model_type": "qwen3"}`, "unsupported Hadamard pack"},
		{"namespace", `{"schema_version": 2, "base_model_type": "qwen3_5", "tensor_namespace": "mlx-lm-text"}`, "tensor namespace"},
		{"ungrouped", `{"schema_version": 2, "base_model_type": "qwen3_5", "tensor_namespace": "mlx-vlm-qwen3_5"}`, "GDN activation layout"},
		{"no modules", `{` + header + `}`, "missing its packed module manifest"},
		{"duplicate", `{` + header + `, "modules": [` + module(`"block": 512, "dtype": "float16"`) + `, ` + module(`"block": 512, "dtype": "float16"`) + `]}`, "duplicate packed module"},
		{"bfloat16", `{` + header + `, "modules": [` + module(`"block": 512, "dtype": "bfloat16"`) + `]}`, "activation dtype"},
		{"block", `{` + header + `, "modules": [` + module(`"block": 256, "dtype": "float16"`) + `]}`, "Hadamard block 256"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parsePrismHadamardConfig([]byte(tt.config)); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// prismTestConfig is a two-layer hybrid stack whose every rotated input width
// is a multiple of the smallest Prism block.
func prismTestConfig() *Config {
	return &Config{
		HiddenSize:            512,
		IntermediateSize:      512,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		NumKeyValueHeads:      1,
		HeadDim:               256,
		RMSNormEps:            1e-6,
		VocabSize:             8,
		TieWordEmbeddings:     true,
		LayerTypes:            []string{"linear_attention", "full_attention"},
		FullAttentionInterval: 2,
		LinearNumKeyHeads:     1,
		LinearNumValueHeads:   2,
		LinearKeyHeadDim:      128,
		LinearValueHeadDim:    256,
		LinearConvKernelDim:   4,
		QuantGroupSize:        128,
		QuantBits:             2,
		QuantMode:             "affine",
	}
}

// prismPack holds a synthetic Prism Hadamard pack in the runner's tensor
// naming. As in real packs, modules of the same input width carry equal signs.
type prismPack struct {
	tensors map[string]*mlx.Array
	modules []hadamardModule
}

func prismSigns(width int) *mlx.Array {
	values := make([]float32, width)
	for i := range values {
		values[i] = 1
		if (i*7)%5 < 2 {
			values[i] = -1
		}
	}
	return mlx.FromValues(values, width)
}

func (p *prismPack) dense(name string, shape ...int) {
	size := 1
	for _, d := range shape {
		size *= d
	}
	values := make([]float32, size)
	for i := range values {
		values[i] = 1 + float32(i%5)/16
	}
	p.tensors["language_model."+name] = mlx.FromValues(values, shape...)
}

func (p *prismPack) packed(path string, rows, cols int, embedding bool) {
	weight, scales, qbiases := mlx.Quantize(patternArray(rows, cols).AsType(mlx.DTypeFloat16), 128, 2, "affine")
	name := "language_model." + path
	p.tensors[name+".weight"] = weight
	p.tensors[name+".weight_scale"] = scales
	p.tensors[name+".weight_qbias"] = qbiases
	p.tensors[name+".signs"] = prismSigns(cols)
	p.modules = append(p.modules, hadamardModule{Path: path, Block: 512, Embedding: embedding, DType: "float16"})
}

func newPrismPack() *prismPack {
	p := &prismPack{tensors: make(map[string]*mlx.Array)}
	p.packed("model.embed_tokens", 8, 512, true)
	p.dense("model.norm.weight", 512)
	for layer, attention := range []string{"linear_attn", "self_attn"} {
		prefix := "model.layers." + strconv.Itoa(layer) + "."
		p.dense(prefix+"input_layernorm.weight", 512)
		p.dense(prefix+"post_attention_layernorm.weight", 512)
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			p.packed(prefix+"mlp."+proj, 512, 512, false)
		}
		if attention == "linear_attn" {
			p.packed(prefix+"linear_attn.in_proj_qkv", 768, 512, false)
			p.packed(prefix+"linear_attn.in_proj_z", 512, 512, false)
			p.packed(prefix+"linear_attn.out_proj", 512, 512, false)
			p.dense(prefix+"linear_attn.in_proj_b.weight", 2, 512)
			p.dense(prefix+"linear_attn.in_proj_a.weight", 2, 512)
			p.dense(prefix+"linear_attn.conv1d.weight", 768, 4, 1)
			p.dense(prefix+"linear_attn.norm.weight", 256)
			p.dense(prefix+"linear_attn.dt_bias", 2)
			p.dense(prefix+"linear_attn.A_log", 2)
		} else {
			p.packed(prefix+"self_attn.q_proj", 1024, 512, false)
			p.packed(prefix+"self_attn.k_proj", 256, 512, false)
			p.packed(prefix+"self_attn.v_proj", 256, 512, false)
			p.packed(prefix+"self_attn.o_proj", 512, 512, false)
			p.dense(prefix+"self_attn.q_norm.weight", 256)
			p.dense(prefix+"self_attn.k_norm.weight", 256)
		}
	}
	return p
}

func (p *prismPack) model() *Model {
	cfg := prismTestConfig()
	return &Model{Config: cfg, Layers: make([]*Layer, cfg.NumHiddenLayers), hadamardModules: p.modules}
}

func TestPrismHadamardLoadWeights(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		p := newPrismPack()
		m := p.model()
		if err := m.LoadWeights(p.tensors); err != nil {
			t.Fatalf("LoadWeights: %v", err)
		}

		if _, ok := m.EmbedTokens.(*nn.HadamardEmbedding); !ok {
			t.Errorf("embedding is %T, want *nn.HadamardEmbedding", m.EmbedTokens)
		}
		if _, ok := m.LMHead.(*nn.HadamardLinear); !ok {
			t.Errorf("tied head is %T, want *nn.HadamardLinear", m.LMHead)
		}
		gdn := m.Layers[0].Linear
		if qkvz, ok := gdn.InProjQKVZ.(*nn.HadamardLinear); !ok {
			t.Errorf("in_proj_qkvz is %T, want *nn.HadamardLinear", gdn.InProjQKVZ)
		} else if got := qkvz.OutputDim(); got != 768+512 {
			t.Errorf("in_proj_qkvz rows = %d, want %d", got, 768+512)
		}
		if ba, ok := gdn.InProjBA.(*nn.Linear); !ok || ba.Weight.DType() != mlx.DTypeFloat16 {
			t.Errorf("in_proj_ba = %T, want a float16 *nn.Linear", gdn.InProjBA)
		}
		if _, ok := m.Layers[1].FullAttn.QProj.(*nn.HadamardLinear); !ok {
			t.Errorf("q_proj is %T, want *nn.HadamardLinear", m.Layers[1].FullAttn.QProj)
		}

		// Activations run in the pack's float16; the decay keeps float32.
		for name, got := range map[string]mlx.DType{
			"final norm":     m.Norm.Weight.DType(),
			"input norm":     m.Layers[0].InputNorm.Weight.DType(),
			"gdn norm":       gdn.NormWeight.DType(),
			"dt_bias":        gdn.DtBias.DType(),
			"attention norm": m.Layers[1].FullAttn.QNorm.Weight.DType(),
		} {
			if got != mlx.DTypeFloat16 {
				t.Errorf("%s dtype = %v, want %v", name, got, mlx.DTypeFloat16)
			}
		}
		if got := gdn.AExp.DType(); got != mlx.DTypeFloat32 {
			t.Errorf("A exp dtype = %v, want %v", got, mlx.DTypeFloat32)
		}
		// Pack norms are stored final; nothing marks them zero-centered.
		if got := m.Norm.Weight.AsType(mlx.DTypeFloat32).Floats()[0]; got != 1 {
			t.Errorf("final norm weight[0] = %v, want 1 (no +1 shift)", got)
		}
	})
}

func TestPrismHadamardLoadWeightsRejectsInvalidPacks(t *testing.T) {
	const upProj = "language_model.model.layers.0.mlp.up_proj"
	for _, tt := range []struct {
		name    string
		mutate  func(*prismPack)
		wantErr string
	}{
		{"missing signs", func(p *prismPack) { delete(p.tensors, upProj+".signs") }, "missing Hadamard signs"},
		{"invalid signs", func(p *prismPack) {
			p.tensors[upProj+".signs"] = mlx.Concatenate([]*mlx.Array{mlx.FromValues([]float32{0}, 1), prismSigns(511)}, 0)
		}, "must be -1 or +1"},
		{"signs width", func(p *prismPack) { p.tensors[upProj+".signs"] = prismSigns(1024) }, "do not fit input width 512"},
		{"stray signs", func(p *prismPack) { p.tensors["language_model.model.norm.signs"] = prismSigns(512) }, "belong to no rotated module"},
		{"unrotated with signs", func(p *prismPack) { p.modules[1].Block = 0 }, "unrotated module carries"},
		{"embedding flag", func(p *prismPack) { p.modules[1].Embedding = true }, "embedding flag"},
		{"not quantized", func(p *prismPack) { delete(p.tensors, upProj+".weight_scale") }, "not quantized"},
		{"output bias", func(p *prismPack) { p.tensors[upProj+".bias"] = prismSigns(512) }, "output bias"},
		{"missing module", func(p *prismPack) {
			p.modules = append(p.modules, hadamardModule{Path: "model.layers.0.mlp.shared_expert.up_proj", Block: 512, DType: "float16"})
		}, "missing weight"},
		{"module never loaded", func(p *prismPack) {
			p.packed("model.layers.0.mlp.shared_expert.up_proj", 512, 512, false)
		}, "is not a projection this model loads"},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			p := newPrismPack()
			tt.mutate(p)
			if err := p.model().LoadWeights(p.tensors); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("LoadWeights error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// A separately imported draft's HF-layout MTP tensors must not make the
// target shift its own, already-final norm weights.
func TestStoresZeroCenteredNormsIgnoresDraftTensors(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tensors := map[string]*mlx.Array{
			"model.norm.weight":         mlx.FromValues([]float32{1}, 1),
			"draft.mtp.fc.weight":       mlx.FromValues([]float32{1}, 1, 1),
			"draft.mtp.norm.weight":     mlx.FromValues([]float32{1}, 1),
			"model.embed_tokens.weight": mlx.FromValues([]float32{1}, 1, 1),
		}
		if (&Model{draftPrefix: "draft."}).storesZeroCenteredNorms(tensors) {
			t.Error("draft tensors marked the target's norms zero-centered")
		}
		tensors["mtp.fc.weight"] = mlx.FromValues([]float32{1}, 1, 1)
		if !(&Model{draftPrefix: "draft."}).storesZeroCenteredNorms(tensors) {
			t.Error("inline MTP tensors no longer mark the norms zero-centered")
		}
	})
}
