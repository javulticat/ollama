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

		embedding, ok := m.EmbedTokens.(*nn.HadamardEmbedding)
		if !ok {
			t.Fatalf("embedding is %T, want *nn.HadamardEmbedding", m.EmbedTokens)
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
		// Projections reading the same input rotate it once, at the layer;
		// down_proj reads the MLP's own activations and keeps its rotation.
		attn := m.Layers[1].FullAttn
		if attn.InputRotation == nil {
			t.Error("attention has no shared input rotation")
		}
		for name, proj := range map[string]nn.LinearLayer{"q_proj": attn.QProj, "k_proj": attn.KProj, "v_proj": attn.VProj} {
			if _, ok := proj.(*nn.QuantizedLinear); !ok {
				t.Errorf("%s is %T, want *nn.QuantizedLinear under the shared rotation", name, proj)
			}
		}
		if _, ok := attn.OProj.(*nn.HadamardLinear); !ok {
			t.Errorf("o_proj is %T, want *nn.HadamardLinear", attn.OProj)
		}
		for i, layer := range m.Layers {
			mlp, ok := layer.MLP.(*DenseMLP)
			if !ok {
				t.Fatalf("layer %d MLP is %T, want *DenseMLP", i, layer.MLP)
			}
			if mlp.InputRotation == nil {
				t.Errorf("layer %d MLP has no shared input rotation", i)
			}
			if _, ok := mlp.DownProj.(*nn.HadamardLinear); !ok {
				t.Errorf("layer %d down_proj is %T, want *nn.HadamardLinear", i, mlp.DownProj)
			}
		}
		// Every input width has one rotation, shared by its modules.
		if embedding.Rotation != attn.InputRotation {
			t.Error("the embedding and attention rotations of the same width are distinct")
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

// Projections whose rotations differ keep their own: only projections sharing
// one rotation may rotate their common input once.
func TestPrismHadamardKeepsDistinctRotations(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		p := newPrismPack()
		for _, proj := range []string{"model.layers.1.self_attn.k_proj", "model.layers.0.mlp.up_proj"} {
			name := "language_model." + proj + ".signs"
			p.tensors[name] = mlx.Neg(p.tensors[name])
		}
		m := p.model()
		if err := m.LoadWeights(p.tensors); err != nil {
			t.Fatalf("LoadWeights: %v", err)
		}

		attn := m.Layers[1].FullAttn
		q, qRotated := attn.QProj.(*nn.HadamardLinear)
		k, kRotated := attn.KProj.(*nn.HadamardLinear)
		switch {
		case attn.InputRotation != nil:
			t.Error("attention shares one rotation although k_proj's differs")
		case !qRotated || !kRotated:
			t.Errorf("q_proj, k_proj = %T, %T, want each to keep its rotation", attn.QProj, attn.KProj)
		case q.Rotation == k.Rotation:
			t.Error("q_proj and k_proj share a rotation although their signs differ")
		}

		mlp0, ok0 := m.Layers[0].MLP.(*DenseMLP)
		mlp1, ok1 := m.Layers[1].MLP.(*DenseMLP)
		if !ok0 || !ok1 {
			t.Fatalf("MLPs are %T, %T, want *DenseMLP", m.Layers[0].MLP, m.Layers[1].MLP)
		}
		if _, ok := mlp0.UpProj.(*nn.HadamardLinear); !ok || mlp0.InputRotation != nil {
			t.Errorf("layer 0 MLP: up_proj = %T, shared rotation = %v; want up_proj to keep its own", mlp0.UpProj, mlp0.InputRotation != nil)
		}
		if mlp1.InputRotation == nil {
			t.Error("layer 1 MLP, whose gate and up match, lost its shared rotation")
		}
	})
}

// A hoisted rotation computes the same MLP as rotating inside each projection.
func TestDenseMLPSharedRotationMatchesPerProjection(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const width, hidden = 512, 512
		rotation := &nn.Hadamard{Signs: prismSigns(width), Block: 512}
		down := &nn.Hadamard{Signs: prismSigns(hidden), Block: 512}
		quantized := func(rows, cols int) nn.LinearLayer {
			return nn.NewQuantizedLinear(patternArray(rows, cols).AsType(mlx.DTypeFloat16), nil, 128, 2, "affine")
		}
		gate, up, downProj := quantized(hidden, width), quantized(hidden, width), quantized(width, hidden)
		x := patternArray(3, width).AsType(mlx.DTypeFloat16)

		shared := &DenseMLP{GateProj: gate, UpProj: up, DownProj: &nn.HadamardLinear{Inner: downProj, Rotation: down}, InputRotation: rotation}
		perProjection := &DenseMLP{
			GateProj: &nn.HadamardLinear{Inner: gate, Rotation: rotation},
			UpProj:   &nn.HadamardLinear{Inner: up, Rotation: rotation},
			DownProj: &nn.HadamardLinear{Inner: downProj, Rotation: down},
		}
		assertBitEqual(t, "mlp", shared.Forward(x, nil), perProjection.Forward(x, nil))
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
