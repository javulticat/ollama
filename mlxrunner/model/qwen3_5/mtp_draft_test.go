package qwen3_5

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestCheckMTPDraftConfig(t *testing.T) {
	target := prismTestConfig()
	draft := *target
	if err := checkMTPDraftConfig(&draft, target); err != nil {
		t.Fatalf("matching draft rejected: %v", err)
	}
	draft.VocabSize++
	if err := checkMTPDraftConfig(&draft, target); err == nil || !strings.Contains(err.Error(), "vocab_size") {
		t.Fatalf("error = %v, want a vocab_size mismatch", err)
	}
}

// mtpDraftTensors builds a single-layer MTP head for prismTestConfig under
// prefix, with its fc quantized to 4 bits as mlx-vlm's drafters ship it.
func mtpDraftTensors(prefix string) (map[string]*mlx.Array, map[string]*model.TensorQuantInfo) {
	tensors := make(map[string]*mlx.Array)
	norm := func(name string, width int) {
		values := make([]float32, width)
		for i := range values {
			values[i] = 1
		}
		tensors[prefix+name] = mlx.FromValues(values, width).AsType(mlx.DTypeFloat16)
	}
	dense := func(name string, rows, cols int) {
		tensors[prefix+name+".weight"] = patternArray(rows, cols).AsType(mlx.DTypeFloat16)
	}

	fc, scales, qbiases := mlx.Quantize(patternArray(512, 1024).AsType(mlx.DTypeFloat16), 64, 4, "affine")
	tensors[prefix+"fc.weight"], tensors[prefix+"fc.weight_scale"], tensors[prefix+"fc.weight_qbias"] = fc, scales, qbiases
	norm("pre_fc_norm_embedding.weight", 512)
	norm("pre_fc_norm_hidden.weight", 512)
	norm("norm.weight", 512)
	norm("layers.0.input_layernorm.weight", 512)
	norm("layers.0.post_attention_layernorm.weight", 512)
	dense("layers.0.self_attn.q_proj", 1024, 512)
	dense("layers.0.self_attn.k_proj", 256, 512)
	dense("layers.0.self_attn.v_proj", 256, 512)
	dense("layers.0.self_attn.o_proj", 512, 512)
	norm("layers.0.self_attn.q_norm.weight", 256)
	norm("layers.0.self_attn.k_norm.weight", 256)
	for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
		dense("layers.0.mlp."+proj, 512, 512)
	}
	return tensors, map[string]*model.TensorQuantInfo{prefix + "fc.weight": {QuantType: "INT4", GroupSize: 64}}
}

func TestMTPDraftModelLoadWeights(t *testing.T) {
	newDraft := func(tensorQuant map[string]*model.TensorQuantInfo) *mtpDraftModel {
		cfg := prismTestConfig()
		cfg.TensorQuant = tensorQuant
		return &mtpDraftModel{target: &Model{Config: cfg}, tensorPrefix: "draft."}
	}

	for _, tt := range []struct {
		name, prefix string
		wantNorm     float32
	}{
		// mlx-vlm drafters store final norms.
		{"mlx-vlm layout", "draft.", 1},
		// A head that keeps its checkpoint's mtp. prefix is zero-centered.
		{"reference layout", "draft.mtp.", 2},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			tensors, tensorQuant := mtpDraftTensors(tt.prefix)
			d := newDraft(tensorQuant)
			if err := d.LoadWeights(tensors); err != nil {
				t.Fatalf("LoadWeights: %v", err)
			}
			fc, ok := d.Head.FC.(*nn.QuantizedLinear)
			if !ok || fc.Bits != 4 || fc.GroupSize != 64 {
				t.Fatalf("fc = %T %+v, want a 4-bit group-64 *nn.QuantizedLinear", d.Head.FC, fc)
			}
			if got := d.Head.Norm.Weight.AsType(mlx.DTypeFloat32).Floats()[0]; got != tt.wantNorm {
				t.Errorf("norm weight[0] = %v, want %v", got, tt.wantNorm)
			}
		})
	}

	mlxtest.RunSubtest(t, "rejects a second layer", func(t *mlxtest.T) {
		tensors, tensorQuant := mtpDraftTensors("draft.")
		tensors["draft.layers.1.input_layernorm.weight"] = tensors["draft.layers.0.input_layernorm.weight"]
		if err := newDraft(tensorQuant).LoadWeights(tensors); err == nil || !strings.Contains(err.Error(), "layer 1") {
			t.Fatalf("LoadWeights error = %v, want a second-layer rejection", err)
		}
	})

	mlxtest.RunSubtest(t, "requires per-tensor quantization", func(t *mlxtest.T) {
		tensors, _ := mtpDraftTensors("draft.")
		if err := newDraft(nil).LoadWeights(tensors); err == nil || !strings.Contains(err.Error(), "no quantization metadata") {
			t.Fatalf("LoadWeights error = %v, want missing quantization metadata", err)
		}
	})
}
