package create

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func specByName(specs []BlobSpec, name string) (BlobSpec, bool) {
	for _, s := range specs {
		if s.Name == name {
			return s, true
		}
	}
	return BlobSpec{}, false
}

func inputByOutput(spec BlobSpec, outputName string) (TensorSpec, bool) {
	for _, ts := range spec.Tensors {
		if ts.Name == outputName {
			return ts, true
		}
	}
	return TensorSpec{}, false
}

// sourceName returns the (single) source tensor name for a TensorSpec.
func sourceName(ts TensorSpec) string {
	if len(ts.Sources) == 0 {
		return ""
	}
	return ts.Sources[0].Name
}

func specNames(specs []BlobSpec) []string {
	names := make([]string, len(specs))
	for i, s := range specs {
		names[i] = s.Name
	}
	return names
}

// setShape overrides the fixture's default shape for one inventory tensor.
func setShape(inv Inventory, name string, shape ...int32) {
	t := inv.Tensors[name]
	t.Shape = shape
	inv.Tensors[name] = t
}

// mlxAffineInventory builds an MLX prequantized layer "l" with 128 inputs
// packed at bits per value in groups of groupSize.
func mlxAffineInventory(q sourceQuantization, bits, groupSize int32) Inventory {
	inv := newInventory(sourceModelConfig{Quantization: q}, map[string]string{
		"l.weight":    "U32",
		"l.scales":    "BF16",
		"l.biases":    "BF16",
		"norm.weight": "BF16",
	})
	setShape(inv, "l.weight", 128, 128*bits/32)
	setShape(inv, "l.scales", 128, 128/groupSize)
	setShape(inv, "l.biases", 128, 128/groupSize)
	return inv
}

func TestPlanPrequantizedMLX(t *testing.T) {
	cfg := sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 32}}
	inv := newInventory(cfg, map[string]string{
		"l.weight":    "U32",
		"l.scales":    "BF16",
		"l.biases":    "BF16",
		"norm.weight": "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	// l.weight (fused with scales+biases) and norm.weight (pass-through).
	if len(specs) != 2 {
		t.Fatalf("got %d specs %v, want 2", len(specs), specNames(specs))
	}

	w, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	for _, want := range []string{"l.weight", "l.weight.scale", "l.weight.bias"} {
		in, ok := inputByOutput(w, want)
		if !ok {
			t.Fatalf("l.weight blob missing input %q", want)
		}
		if in.Transform != TransformNone {
			t.Errorf("%s transform = %q, want none", want, in.Transform)
		}
	}
	if w.Metadata["quant_type"] != "int4" || w.Metadata["group_size"] != "32" {
		t.Errorf("metadata = %v, want quant_type=int4 group_size=32 from config", w.Metadata)
	}
	if _, ok := specByName(specs, "norm.weight"); !ok {
		t.Error("norm.weight should pass through as its own blob")
	}
}

func TestPlanPrequantizedFlintEmbeddingBits(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want string
	}{
		{name: "int8 embedding", raw: `{"mlx_export":{"embedding_bits":8}}`, want: "int8"},
		{name: "int4 embedding", raw: `{"mlx_export":{"embedding_bits":4}}`, want: "int4"},
		{name: "missing bit width", raw: `{"mlx_export":{}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := newInventory(sourceModelConfig{
				Architectures: []string{"FlintForCausalLM"},
				Quantization:  sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 64},
			}, map[string]string{
				"model.embed_tokens.weight":              "U32",
				"model.embed_tokens.scales":              "BF16",
				"model.embed_tokens.biases":              "BF16",
				"model.layers.0.self_attn.q_proj.weight": "U32",
				"model.layers.0.self_attn.q_proj.scales": "BF16",
				"model.layers.0.self_attn.q_proj.biases": "BF16",
			})
			inv.RawConfig = []byte(tc.raw)
			specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
			if tc.want == "" {
				if err == nil || !strings.Contains(err.Error(), "mlx_export.embedding_bits") {
					t.Fatalf("Plan() error = %v, want missing embedding bit width", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			embedding, ok := specByName(specs, "model.embed_tokens.weight")
			if !ok || embedding.Metadata["quant_type"] != tc.want {
				t.Errorf("embedding quant_type = %q, want %q", embedding.Metadata["quant_type"], tc.want)
			}
			projection, ok := specByName(specs, "model.layers.0.self_attn.q_proj.weight")
			if !ok || projection.Metadata["quant_type"] != "int4" {
				t.Errorf("projection quant_type = %q, want int4", projection.Metadata["quant_type"])
			}
		})
	}
}

func TestPlanPrequantizedMLXModuleQuantization(t *testing.T) {
	int4 := map[string]string{"quant_type": "int4", "group_size": "64"}
	for _, tt := range []struct {
		name         string
		q            sourceQuantization
		want         map[string]string // the gate's metadata
		wantFileType string
		wantErr      string
	}{
		{
			// mlx-lm keeps Qwen3-Next's MoE routers at 8 bits.
			name:         "8-bit router",
			q:            sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Bits: 8, GroupSize: 64}}},
			want:         map[string]string{"quant_type": "int8", "group_size": "64"},
			wantFileType: "int4",
		},
		{
			name:         "restates the default",
			q:            sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Bits: 4, GroupSize: 64, Mode: "affine"}}},
			want:         int4,
			wantFileType: "int4",
		},
		{
			name:         "omitted fields take MLX's defaults, not the config's",
			q:            sourceQuantization{Bits: 8, GroupSize: 32, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Mode: "affine"}}},
			want:         int4,
			wantFileType: "int8",
		},
		{
			name:         "affine router under an mxfp4 default",
			q:            sourceQuantization{Bits: 4, GroupSize: 32, Mode: "mxfp4", Modules: map[string]moduleQuantization{"mlp.gate": {Bits: 8, GroupSize: 64}}},
			want:         map[string]string{"quant_type": "int8", "group_size": "64"},
			wantFileType: "mxfp4",
		},
		{
			name:         "mxfp8 module under an affine default",
			q:            sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Bits: 8, Mode: "mxfp8"}}},
			want:         map[string]string{"quant_type": "mxfp8", "group_size": "32"},
			wantFileType: "int4",
		},
		{
			// mxfp8 fixes its width, so an entry may omit the bits.
			name:         "mxfp8 module without bits",
			q:            sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Mode: "mxfp8"}}},
			want:         map[string]string{"quant_type": "mxfp8", "group_size": "32"},
			wantFileType: "int4",
		},
		{
			// An entry no weight uses has no effect, whatever its type.
			name:         "unused entry",
			q:            sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"vision_tower.blocks.0.attn.qkv": {Bits: 7}}},
			want:         int4,
			wantFileType: "int4",
		},
		{
			// mlx-lm omits the mode for affine, and its entries apply the same.
			name:         "default without a mode",
			q:            sourceQuantization{Bits: 4, GroupSize: 64, Modules: map[string]moduleQuantization{"mlp.gate": {Bits: 8, GroupSize: 64}}},
			want:         map[string]string{"quant_type": "int8", "group_size": "64"},
			wantFileType: "int4",
		},
		{
			name:    "unsupported width",
			q:       sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Bits: 7}}},
			wantErr: "mlp.gate: config.json quantizes this module with 7 bits, which is not supported",
		},
		{
			name:    "unreadable entry",
			q:       sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Unreadable: true}}},
			wantErr: "mlp.gate: config.json's quantization entry for this module has unreadable bits, group_size or mode",
		},
		{
			name:    "unknown mode",
			q:       sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: map[string]moduleQuantization{"mlp.gate": {Mode: "fp4"}}},
			wantErr: `mlp.gate: config.json quantizes this module with mode "fp4", which is not supported`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			inv := newInventory(sourceModelConfig{Quantization: tt.q}, map[string]string{
				"mlp.gate.weight":    "U32",
				"mlp.gate.scales":    "BF16",
				"mlp.gate.biases":    "BF16",
				"mlp.up_proj.weight": "U32",
				"mlp.up_proj.scales": "BF16",
				"mlp.up_proj.biases": "BF16",
				"norm.weight":        "BF16",
			})
			class, err := Classify(inv, "")
			if err != nil {
				t.Fatalf("Classify() error = %v", err)
			}
			specs, err := Plan(inv, class, defaultQuantPolicy{})
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Plan() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Plan() error = %v", err)
			}

			gate, ok := specByName(specs, "mlp.gate.weight")
			if !ok || len(gate.Tensors) != 3 {
				t.Fatalf("mlp.gate.weight blob = %+v, want the weight fused with its scales and biases", gate)
			}
			if !maps.Equal(gate.Metadata, tt.want) {
				t.Errorf("gate metadata = %v, want %v", gate.Metadata, tt.want)
			}
			// A weight without an entry keeps the default.
			up, ok := specByName(specs, "mlp.up_proj.weight")
			if !ok {
				t.Fatal("missing mlp.up_proj.weight blob")
			}
			if want := inv.Config.QuantMetadata(); !maps.Equal(up.Metadata, want) {
				t.Errorf("up_proj metadata = %v, want the default %v", up.Metadata, want)
			}
			if class.Quantize != tt.wantFileType {
				t.Errorf("file type = %q, want %q", class.Quantize, tt.wantFileType)
			}
		})
	}
}

// TestPlanPrequantizedMLXModuleQuantizationFromConfig runs a config as mlx-lm
// writes it, with the entries in both quantization blocks, through import.
func TestPlanPrequantizedMLXModuleQuantizationFromConfig(t *testing.T) {
	const quantization = `{
		"group_size": 64, "bits": 4, "mode": "affine",
		"language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
		"language_model.model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8}
	}`
	var cfg sourceModelConfig
	if err := json.Unmarshal([]byte(`{"quantization": `+quantization+`, "quantization_config": `+quantization+`}`), &cfg); err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]string)
	for _, module := range []string{"mlp.gate", "mlp.shared_expert_gate", "mlp.switch_mlp.up_proj"} {
		base := "language_model.model.layers.0." + module
		tensors[base+".weight"] = "U32"
		tensors[base+".scales"] = "BF16"
		tensors[base+".biases"] = "BF16"
	}
	inv := newInventory(cfg, tensors)

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	for module, want := range map[string]string{
		"mlp.gate":               "int8",
		"mlp.shared_expert_gate": "int8",
		"mlp.switch_mlp.up_proj": "int4",
	} {
		name := "language_model.model.layers.0." + module + ".weight"
		spec, ok := specByName(specs, name)
		if !ok {
			t.Fatalf("missing %s blob", name)
		}
		if got := spec.Metadata; got["quant_type"] != want || got["group_size"] != "64" {
			t.Errorf("%s metadata = %v, want %s with group size 64", name, got, want)
		}
	}
}

// TestPlanPrequantizedModelOptIgnoresModuleEntries checks that per-module
// entries apply only to MLX weights: one on a ModelOpt weight is neither
// checked nor recorded, and does not change the file type.
func TestPlanPrequantizedModelOptIgnoresModuleEntries(t *testing.T) {
	cfg := sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", Modules: map[string]moduleQuantization{"l": {Bits: 7}}}}
	inv := newInventory(cfg, map[string]string{
		"l.weight":         "U8",
		"l.weight_scale":   "F8_E4M3",
		"l.weight_scale_2": "F32",
	})

	class, err := Classify(inv, "")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if class.Quantize != "nvfp4" {
		t.Errorf("file type = %q, want nvfp4", class.Quantize)
	}
	specs, err := Plan(inv, class, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	l, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	if want := map[string]string{"quant_type": "nvfp4"}; !maps.Equal(l.Metadata, want) {
		t.Errorf("metadata = %v, want %v", l.Metadata, want)
	}
}

// The file type is the default when a weight keeps it. mlx-lm's mixed recipes
// give every weight an entry, and then the most common type names the model.
func TestClassifyModuleEntriesFileType(t *testing.T) {
	for _, tt := range []struct {
		name    string
		modules map[string]moduleQuantization
		want    string
	}{
		{"a weight keeps the default", map[string]moduleQuantization{"a": {Bits: 8, GroupSize: 64}}, "int4"},
		{"every weight has an entry", map[string]moduleQuantization{"a": {Bits: 8, GroupSize: 64}, "b": {Bits: 8, GroupSize: 64}, "c": {Mode: "mxfp8"}}, "int8"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			inv := newInventory(sourceModelConfig{Quantization: sourceQuantization{Bits: 4, GroupSize: 64, Mode: "affine", Modules: tt.modules}}, map[string]string{
				"a.weight": "U32", "a.scales": "BF16", "a.biases": "BF16",
				"b.weight": "U32", "b.scales": "BF16", "b.biases": "BF16",
				"c.weight": "U32", "c.scales": "BF16", "c.biases": "BF16",
			})
			if got := detectPrequantizedQuantization(inv); got != tt.want {
				t.Errorf("file type = %q, want %q", got, tt.want)
			}
		})
	}
}

// Entries mean nothing unless the default names an MLX type, so another
// producer's config is neither checked nor honored.
func TestPlanPrequantizedIgnoresEntriesOutsideMLX(t *testing.T) {
	cfg := sourceModelConfig{Quantization: sourceQuantization{Bits: 4, GroupSize: 128, QuantMethod: "gptq", Modules: map[string]moduleQuantization{"l": {Bits: 7}}}}
	inv := newInventory(cfg, map[string]string{"l.weight": "U32", "l.scales": "BF16", "l.biases": "BF16"})
	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	l, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	if l.Metadata != nil {
		t.Errorf("metadata = %v, want none", l.Metadata)
	}
}

func TestPlanPrequantizedModelOptNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":         "U8",
		"l.weight_scale":   "F8_E4M3",
		"l.weight_scale_2": "F32",
	})
	global := inv.Tensors["l.weight_scale_2"]
	global.Shape = nil
	inv.Tensors[global.Name] = global

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1", len(specs), specNames(specs))
	}
	w := specs[0]
	if w.Name != "l.weight" {
		t.Fatalf("blob name = %q, want l.weight", w.Name)
	}

	weightIn, _ := inputByOutput(w, "l.weight")
	if weightIn.Transform != TransformRepackFP4 || weightIn.OutDtype != "U32" || !slices.Equal(weightIn.OutShape, []int32{128, 32}) {
		t.Errorf("weight input = %+v, want repack to U32 [128 32]", weightIn)
	}
	scaleIn, _ := inputByOutput(w, "l.weight.scale")
	if scaleIn.Transform != TransformRelabelU8 || scaleIn.OutDtype != "U8" {
		t.Errorf("scale input = %+v, want relabel to U8", scaleIn)
	}
	globalIn, ok := inputByOutput(w, "l.weight.global_scale")
	if !ok || globalIn.Transform != TransformScalarF32 {
		t.Errorf("global_scale input = %+v ok=%v, want scalar_f32 (stored as-is)", globalIn, ok)
	}
	if w.Metadata["quant_type"] != "nvfp4" {
		t.Errorf("quant_type = %q, want nvfp4", w.Metadata["quant_type"])
	}
	if _, ok := w.Metadata["group_size"]; ok {
		t.Errorf("ModelOpt should not default group_size: %v", w.Metadata)
	}
}

func TestPlanPrequantizedModelOptExpertBank(t *testing.T) {
	inv := newInventory(sourceModelConfig{Architectures: []string{"FlintForCausalLM"}}, map[string]string{
		"model.layers.2.mlp.experts.0.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale":   "U8",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale_2": "F32",
		"model.layers.2.mlp.experts.1.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.1.gate_proj.weight_scale":   "U8",
		"model.layers.2.mlp.experts.1.gate_proj.weight_scale_2": "F32",
	})
	for name, tensor := range inv.Tensors {
		if slices.Contains([]string{"model.layers.2.mlp.experts.0.gate_proj.weight_scale_2", "model.layers.2.mlp.experts.1.gate_proj.weight_scale_2"}, name) {
			tensor.Shape = []int32{1}
		}
		inv.Tensors[name] = tensor
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got blobs %v, want one expert bank", specNames(specs))
	}
	name := "model.layers.2.mlp.experts.gate_proj.weight"
	bank := specs[0]
	if bank.Name != "model.layers.2.mlp.experts" || bank.Metadata["quant_type"] != "nvfp4" {
		t.Fatalf("expert bank = %+v", bank)
	}
	for _, tc := range []struct {
		name  string
		shape []int32
	}{
		{name, []int32{2, 128, 128}},
		{name + ".scale", []int32{2, 128, 128}},
		{name + ".global_scale", []int32{2}},
	} {
		tensor, ok := inputByOutput(bank, tc.name)
		if !ok || tensor.Transform != TransformStackExperts || !slices.Equal(tensor.OutShape, tc.shape) || len(tensor.Sources) != 2 {
			t.Errorf("%s = %+v, want stacked %v", tc.name, tensor, tc.shape)
		}
	}
}

func TestPlanPrequantizedModelOptExpertBankRejectsIncompleteScale(t *testing.T) {
	inv := newInventory(sourceModelConfig{Architectures: []string{"FlintForCausalLM"}}, map[string]string{
		"model.layers.2.mlp.experts.0.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale":   "U8",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale_2": "F32",
		"model.layers.2.mlp.experts.1.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.1.gate_proj.weight_scale":   "U8",
	})
	global := inv.Tensors["model.layers.2.mlp.experts.0.gate_proj.weight_scale_2"]
	global.Shape = []int32{1}
	inv.Tensors[global.Name] = global

	_, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err == nil || !strings.Contains(err.Error(), "model.layers.2.mlp.experts") {
		t.Fatalf("Plan() error = %v, want named incomplete expert bank", err)
	}
}

func TestPlanPrequantizedModelOptMixedPrecision(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"fp4.weight":         "U8",
		"fp4.weight_scale":   "F8_E4M3",
		"fp4.weight_scale_2": "F32",
		"fp8.weight":         "F8_E4M3",
		"fp8.weight_scale":   "U8",
	})
	classification, err := Classify(inv, "")
	if err != nil || classification.Kind != SourcePrequantized || classification.Quantize != "" {
		t.Fatalf("mixed ModelOpt classification = %+v, %v", classification, err)
	}

	specs, err := Plan(inv, classification, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 2 {
		t.Fatalf("got %d specs %v, want two fused weights", len(specs), specNames(specs))
	}
	fp4, _ := specByName(specs, "fp4.weight")
	if fp4.Metadata["quant_type"] != "nvfp4" {
		t.Errorf("FP4 metadata = %v", fp4.Metadata)
	}
	fp8, _ := specByName(specs, "fp8.weight")
	if fp8.Metadata["quant_type"] != "mxfp8" || fp8.Metadata["group_size"] != "32" {
		t.Errorf("FP8 metadata = %v", fp8.Metadata)
	}
	weight, _ := inputByOutput(fp8, "fp8.weight")
	scale, _ := inputByOutput(fp8, "fp8.weight.scale")
	if weight.Transform != TransformRepackFP8 || weight.OutDtype != "U32" || !slices.Equal(weight.OutShape, []int32{128, 32}) || scale.Transform != TransformNone || scale.OutDtype != "" {
		t.Errorf("FP8 weight must pack into U32 and exponent scale must keep source bytes: weight=%+v scale=%+v", weight, scale)
	}
	if q := detectPrequantizedQuantization(inv); q != "" {
		t.Errorf("mixed precision classification = %q, want no single file type", q)
	}
}

func TestClassifyModelOptMXFP8WithoutNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":       "F8_E4M3",
		"l.weight_scale": "U8",
	})
	classification, err := Classify(inv, "")
	if err != nil || classification.Kind != SourcePrequantized || classification.Quantize != "mxfp8" {
		t.Fatalf("ModelOpt MXFP8 classification = %+v, %v", classification, err)
	}
	specs, err := Plan(inv, classification, defaultQuantPolicy{})
	if err != nil || len(specs) != 1 || specs[0].Metadata["quant_type"] != "mxfp8" {
		t.Fatalf("ModelOpt MXFP8 plan = %+v, %v", specs, err)
	}
}

func TestPlanPrequantizedModelOptPackedNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":         "U32",
		"l.weight_scale":   "F8_E4M3",
		"l.weight_scale_2": "F32",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want one fused weight", len(specs), specNames(specs))
	}
	weight, ok := inputByOutput(specs[0], "l.weight")
	if !ok || weight.Transform != TransformNone || weight.OutDtype != "" || len(weight.OutShape) != 0 {
		t.Errorf("already-packed U32 weight must keep its source shape: %+v ok=%v", weight, ok)
	}
	if specs[0].Metadata["quant_type"] != "nvfp4" {
		t.Errorf("quant_type = %q, want nvfp4", specs[0].Metadata["quant_type"])
	}
}

func TestPlanPrequantizedModelOptDropsActivationScale(t *testing.T) {
	// ModelOpt ships per-weight activation scales (.input_scale and, in some
	// variants, .input_global_scale) that are unused for weight-only
	// inference. They must be consumed, not emitted as their own blobs.
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":             "U8",
		"l.weight_scale":       "F8_E4M3",
		"l.weight_scale_2":     "F32",
		"l.input_scale":        "F32",
		"l.input_global_scale": "F32",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1 (activation scales must not become blobs)", len(specs), specNames(specs))
	}
	w := specs[0]
	for _, act := range []string{"l.input_scale", "l.input_global_scale"} {
		if _, leaked := inputByOutput(w, act); leaked {
			t.Errorf("activation scale %s leaked into the fused blob", act)
		}
		for _, s := range specs {
			if s.Name == act {
				t.Errorf("activation scale %s emitted as its own blob", act)
			}
		}
	}
}

func TestPlanPrequantizedCompressedNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight_packed":       "U8",
		"l.weight_scale":        "F8_E4M3",
		"l.weight_global_scale": "F32",
		"l.input_global_scale":  "F32",
		"l.input_scale":         "F32",
	})
	for _, name := range []string{"l.weight_global_scale", "l.input_global_scale", "l.input_scale"} {
		scale := inv.Tensors[name]
		scale.Shape = nil
		inv.Tensors[name] = scale
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1", len(specs), specNames(specs))
	}
	w := specs[0]
	if w.Name != "l.weight" {
		t.Fatalf("blob name = %q, want l.weight", w.Name)
	}

	weightIn, _ := inputByOutput(w, "l.weight")
	if sourceName(weightIn) != "l.weight_packed" || weightIn.Transform != TransformRepackFP4 {
		t.Errorf("weight input = %+v, want source l.weight_packed repacked", weightIn)
	}
	globalIn, ok := inputByOutput(w, "l.weight.global_scale")
	if !ok || globalIn.Transform != TransformReciprocalF32 {
		t.Errorf("global_scale input = %+v ok=%v, want reciprocal_f32", globalIn, ok)
	}
	if len(w.Tensors) != 3 {
		t.Errorf("fused tensors = %v, want only weight, block scale and global scale", w.Tensors)
	}
	if w.Metadata["quant_type"] != "nvfp4" || w.Metadata["group_size"] != "16" {
		t.Errorf("metadata = %v, want quant_type=nvfp4 group_size=16", w.Metadata)
	}
}

func TestPlanPrequantizedModelOptRetainsExpertScaleBanks(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":         "U32",
		"l.weight_scale":   "U8",
		"l.weight_scale_2": "F32",
		"l.input_scale":    "F32",
	})
	for _, name := range []string{"l.weight_scale_2", "l.input_scale"} {
		scale := inv.Tensors[name]
		scale.Shape = []int32{128}
		inv.Tensors[name] = scale
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	w, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	scale, ok := inputByOutput(w, "l.weight.global_scale")
	if !ok || scale.Transform != TransformF32 || !slices.Equal(scale.Sources[0].Shape, []int32{128}) {
		t.Errorf("global scale = %+v ok=%v, want F32 expert scale bank", scale, ok)
	}
	if len(specs) != 1 || len(w.Tensors) != 3 {
		t.Errorf("specs = %v, want only fused weight, block scale and global scale", specs)
	}
}

func TestPlanPrequantizedRetainsKVCacheScales(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.k_proj.weight":       "U8",
		"l.k_proj.weight_scale": "F8_E4M3",
		"l.k_proj.k_scale":      "F32",
		"l.v_proj.weight":       "U8",
		"l.v_proj.weight_scale": "F8_E4M3",
		"l.v_proj.v_scale":      "F32",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	for _, name := range []string{"l.k_proj.k_scale", "l.v_proj.v_scale"} {
		spec, ok := specByName(specs, name)
		if !ok || len(spec.Tensors) != 1 || sourceName(spec.Tensors[0]) != name {
			t.Errorf("%s = %+v ok=%v, want unchanged standalone tensor", name, spec, ok)
		}
	}
}

// mlx-lm releases from before the "mode" field write only bits and group_size
// for affine weights. The loader cannot tell 4-bit
// group-64 from 8-bit group-32 by shape, and it reads 4-bit group-128 as 8-bit
// group-64, so the blob must record the type.
func TestPlanPrequantizedMLXWithoutMode(t *testing.T) {
	tests := []struct {
		name          string
		q             sourceQuantization
		wantQuant     string
		wantGroupSize string
	}{
		{"4-bit group-64", sourceQuantization{Bits: 4, GroupSize: 64}, "int4", "64"},
		{"4-bit group-128", sourceQuantization{Bits: 4, GroupSize: 128}, "int4", "128"},
		{"8-bit group-32", sourceQuantization{Bits: 8, GroupSize: 32}, "int8", "32"},
		// Other producers name themselves with quant_method, and their bits
		// alone do not imply MLX affine packing.
		{"with quant_method", sourceQuantization{Bits: 4, GroupSize: 64, QuantMethod: "awq"}, "", ""},
		{"unsupported width", sourceQuantization{Bits: 7, GroupSize: 64}, "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inv := mlxAffineInventory(tt.q, int32(tt.q.Bits), int32(tt.q.GroupSize))
			specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
			if err != nil {
				t.Fatalf("Plan() error = %v", err)
			}
			w, ok := specByName(specs, "l.weight")
			if !ok {
				t.Fatal("missing l.weight blob")
			}
			if w.Metadata["quant_type"] != tt.wantQuant || w.Metadata["group_size"] != tt.wantGroupSize {
				t.Errorf("metadata = %v, want quant_type=%q group_size=%q", w.Metadata, tt.wantQuant, tt.wantGroupSize)
			}
		})
	}
}

// mlx-lm's AWQ tool writes a placeholder default and an entry, or false, for
// every leaf module, so each weight imports at its own entry's layout.
func TestPlanPrequantizedMLXAWQConfig(t *testing.T) {
	for _, groupSize := range []int{64, 32} {
		t.Run(fmt.Sprintf("group size %d", groupSize), func(t *testing.T) {
			var cfg sourceModelConfig
			if err := json.Unmarshal(fmt.Appendf(nil, `{"quantization": {
				"group_size": 64, "bits": 4,
				"model.embed_tokens": {"group_size": 32, "bits": 4},
				"model.layers.0.mlp.down_proj": {"group_size": %d, "bits": 4},
				"model.layers.0.input_layernorm": false
			}}`, groupSize), &cfg); err != nil {
				t.Fatal(err)
			}
			inv := newInventory(cfg, map[string]string{
				"model.embed_tokens.weight":             "U32",
				"model.embed_tokens.scales":             "BF16",
				"model.embed_tokens.biases":             "BF16",
				"model.layers.0.mlp.down_proj.weight":   "U32",
				"model.layers.0.mlp.down_proj.scales":   "BF16",
				"model.layers.0.mlp.down_proj.biases":   "BF16",
				"model.layers.0.input_layernorm.weight": "BF16",
			})
			specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
			if err != nil {
				t.Fatalf("Plan() error = %v", err)
			}
			for name, want := range map[string]map[string]string{
				"model.embed_tokens.weight":           {"quant_type": "int4", "group_size": "32"},
				"model.layers.0.mlp.down_proj.weight": {"quant_type": "int4", "group_size": strconv.Itoa(groupSize)},
			} {
				if spec, ok := specByName(specs, name); !ok || !maps.Equal(spec.Metadata, want) {
					t.Errorf("%s metadata = %v, want %v", name, spec.Metadata, want)
				}
			}
		})
	}
}
