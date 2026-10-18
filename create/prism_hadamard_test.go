package create

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

// A Prism Hadamard pack (Ternary Bonsai 2) names only a model_type, stores its
// projections as 2-bit affine weights with a ±1 signs vector per module, and
// carries the rotation contract in hadamard.json.
func TestCreatePrismHadamardPack(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{
		"model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5",
		"text_config": {"model_type": "qwen3_5_text", "hidden_size": 256},
		"vision_config": {},
		"quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
		"modules": [{"path": "model.layers.0.mlp.up_proj", "block": 256, "embedding": false, "dtype": "float16"}],
		"hadamard_config": "hadamard.json"}`)
	if err := os.WriteFile(filepath.Join(dir, "hadamard.json"), []byte(`{"prism.hadamard.version": 1}`), 0o600); err != nil {
		t.Fatal(err)
	}
	const proj = "language_model.model.layers.0.mlp.up_proj"
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		// 256 inputs at 2 bits pack into 16 words, in 2 groups of 128.
		st.NewTensorDataFromBytes(proj+".weight", "U32", []int32{4, 16}, make([]byte, 4*16*4)),
		st.NewTensorDataFromBytes(proj+".scales", "F16", []int32{4, 2}, make([]byte, 4*2*2)),
		st.NewTensorDataFromBytes(proj+".biases", "F16", []int32{4, 2}, make([]byte, 4*2*2)),
		st.NewTensorDataFromBytes(proj+".signs", "F32", []int32{256}, make([]byte, 256*4)),
		st.NewTensorDataFromBytes("language_model.model.norm.weight", "F32", []int32{256}, make([]byte, 256*4)),
	})

	store := newCaptureStore()
	var result ManifestInfo
	err := Create(t.Context(), "bonsai-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, info ManifestInfo) error {
		result = info
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}

	if result.Class.Quantize != "int2" {
		t.Errorf("file type = %q, want int2", result.Class.Quantize)
	}
	if result.ModelConfig.Renderer != "qwen3.5" || result.ModelConfig.Parser != "qwen3.5" {
		t.Errorf("renderer/parser = %q/%q, want qwen3.5", result.ModelConfig.Renderer, result.ModelConfig.Parser)
	}

	weight, ok := store.blobs[proj+".weight"]
	if !ok {
		t.Fatalf("missing fused weight blob; blobs = %v", store.names())
	}
	if md := blobMetadata(t, weight); md["quant_type"] != "int2" || md["group_size"] != "128" {
		t.Errorf("weight metadata = %v, want quant_type=int2 group_size=128", md)
	}
	header := blobHeader(t, weight)
	for _, name := range []string{proj + ".weight", proj + ".weight.scale", proj + ".weight.bias"} {
		if _, ok := header[name]; !ok {
			t.Errorf("fused weight blob is missing %s", name)
		}
	}

	// The signs pass through at source precision for the runner to apply.
	signs, ok := store.blobs[proj+".signs"]
	if !ok {
		t.Fatalf("missing signs blob; blobs = %v", store.names())
	}
	if e := blobHeader(t, signs)[proj+".signs"]; e.Dtype != "F32" || !slices.Equal(e.Shape, []int32{256}) {
		t.Errorf("signs = %+v, want F32 [256]", e)
	}
	if _, ok := store.blobs["hadamard.json"]; !ok {
		t.Error("hadamard.json was not imported")
	}
}
