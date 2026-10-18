package create

import (
	"cmp"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx/quant"
)

// prequantPattern describes how one producer packs an already-quantized weight
// and its scale companions into safetensors files, and how to fuse them into
// the single blob our loader reads. Producers differ only in tensor names and a
// few per-field transforms; expressing them as table rows keeps those
// differences visible and prevents the per-producer drift the old separate code
// paths suffered (for example the global scale being stored as-is by one
// producer and inverted by another).
//
// All suffixes are relative to the base — the source weight name minus its
// weight suffix. The fused blob is always named "<base>.weight", with
// companions "<base>.weight.scale", ".bias", and ".global_scale".
type prequantPattern struct {
	name string

	weightSuffix string   // source suffix identifying the weight (".weight" or ".weight_packed")
	weightDtypes []string // accepted source dtypes; empty accepts any dtype
	scaleDtypes  []string // accepted scale dtypes; empty accepts any dtype
	repackWeight bool     // reinterpret four one-byte weight values as one U32 word

	scaleSuffix    string // required per-block / affine scale companion
	scaleRelabelU8 bool   // relabel an F8_E4M3 scale as U8 for the loader

	biasSuffix string // optional bias / zero-point companion ("" if none)

	globalSuffix     string // optional global-scale companion ("" if none)
	globalReciprocal bool   // producer stores the weight global scale inverted

	ignoreSuffixes []string // activation scales unused by weight-only inference

	forceQuantType   string // override the blob's quant_type metadata
	defaultGroupSize string // set group_size metadata only when the config did not
	moduleEntries    bool   // honor config.json's mlx-lm per-module entries
}

// prequantPatterns is consulted in order; the first whose weight suffix matches
// and whose required scale companion is present wins. MLX and ModelOpt both use
// a ".weight" weight, but their scale companions (".scales" vs ".weight_scale")
// are mutually exclusive, so the order between them does not matter.
var prequantPatterns = []prequantPattern{
	{
		name:          "mlx",
		weightSuffix:  ".weight",
		scaleSuffix:   ".scales",
		biasSuffix:    ".biases",
		moduleEntries: true,
	},
	{
		name:             "compressed-tensors-nvfp4",
		weightSuffix:     ".weight_packed",
		repackWeight:     true,
		scaleSuffix:      ".weight_scale",
		scaleRelabelU8:   true,
		globalSuffix:     ".weight_global_scale",
		globalReciprocal: true,
		ignoreSuffixes:   []string{".input_scale", ".input_global_scale"},
		forceQuantType:   "nvfp4",
		defaultGroupSize: "16",
	},
	{
		name:             "modelopt-mxfp8",
		weightSuffix:     ".weight",
		weightDtypes:     []string{"F8_E4M3", "F8_E4M3FN"},
		repackWeight:     true,
		scaleSuffix:      ".weight_scale",
		scaleDtypes:      []string{"U8"},
		forceQuantType:   "mxfp8",
		defaultGroupSize: "32",
	},
	{
		name:           "modelopt-nvfp4",
		weightSuffix:   ".weight",
		weightDtypes:   []string{"U8", "U32"},
		repackWeight:   true,
		scaleSuffix:    ".weight_scale",
		scaleRelabelU8: true,
		globalSuffix:   ".weight_scale_2",
		ignoreSuffixes: []string{".input_scale", ".input_global_scale"},
		forceQuantType: "nvfp4",
	},
}

// planPrequantized plans an already-quantized source: each weight is fused with
// its scale companions into one blob, companions are not emitted on their own,
// and any remaining tensors (norms, embeddings) pass through at source
// precision.
func planPrequantized(inv Inventory) ([]BlobSpec, error) {
	if err := checkModuleTypes(inv); err != nil {
		return nil, err
	}
	fused := make(map[string]BlobSpec)
	consumed := make(map[string]bool)
	for _, name := range sortedTensorNames(inv) {
		spec, sources, ok := matchPrequant(name, inv)
		if !ok {
			continue
		}
		fused[name] = spec
		for _, s := range sources {
			consumed[s] = true
		}
	}
	if slices.Contains(inv.Config.Architectures, "FlintForCausalLM") {
		const embedding = "model.embed_tokens.weight"
		if spec, ok := fused[embedding]; ok && spec.Metadata["quant_type"] == "int4" {
			var config struct {
				MLXExport struct {
					EmbeddingBits int `json:"embedding_bits"`
				} `json:"mlx_export"`
			}
			if err := json.Unmarshal(inv.RawConfig, &config); err != nil {
				return nil, fmt.Errorf("Flint embedding quantization: %w", err)
			}
			// The packed embedding has the same shape at INT8/group64 as at
			// INT4/group128; only the publisher's export metadata identifies it.
			bits := config.MLXExport.EmbeddingBits
			if bits != 4 && bits != 8 {
				return nil, fmt.Errorf("Flint embedding quantization: unsupported mlx_export.embedding_bits %d", bits)
			}
			if bits == 8 {
				spec.Metadata["quant_type"] = "int8"
			}
			fused[embedding] = spec
		}
	}

	// Check each blob against its final metadata, after Flint's correction.
	for _, name := range sortedKeys(fused) {
		if err := checkAffineWidth(fused[name]); err != nil {
			return nil, err
		}
		if err := checkExpertWidth(fused[name], inv.Config.Architecture()); err != nil {
			return nil, err
		}
	}

	// Model Optimizer keeps Flint's publisher-native per-expert tensor names.
	// As with float imports, the Ollama artifact needs one bank per projection.
	// Stack its packed words and calibrated scales without requantizing them.
	expertGroups := make(map[string][]SourceTensor)
	if slices.Contains(inv.Config.Architectures, "FlintForCausalLM") {
		for _, name := range sortedTensorNames(inv) {
			if group, ok := perExpertGroup(name); ok && strings.HasSuffix(name, ".weight") {
				expertGroups[group] = append(expertGroups[group], inv.Tensors[name])
			}
		}
	}
	var specs []BlobSpec
	for _, group := range sortedKeys(expertGroups) {
		experts := expertGroups[group]
		quantized := 0
		for _, weight := range experts {
			if _, ok := fused[weight.Name]; ok {
				quantized++
			}
		}
		if quantized != 0 && quantized != len(experts) {
			return nil, fmt.Errorf("prequantized expert group %s has %d of %d projections with scales", group, quantized, len(experts))
		}
		var groupSpecs []BlobSpec
		var err error
		if quantized == 0 {
			groupSpecs, err = planExpertGroup(group, experts, "", defaultQuantPolicy{})
		} else {
			groupSpecs, err = planPrequantExpertGroup(group, experts, fused)
		}
		if err != nil {
			return nil, err
		}
		specs = append(specs, groupSpecs...)
		for _, weight := range experts {
			consumed[weight.Name] = true
		}
	}

	for _, name := range sortedTensorNames(inv) {
		if consumed[name] {
			continue
		}
		if spec, ok := fused[name]; ok {
			specs = append(specs, spec)
			continue
		}
		t := inv.Tensors[name]
		specs = append(specs, BlobSpec{Name: name, Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{t}}}})
	}
	return specs, nil
}

func planPrequantExpertGroup(group string, weights []SourceTensor, fused map[string]BlobSpec) ([]BlobSpec, error) {
	// The float importer's expert planner validates contiguous indices. Stacking
	// copies the source bytes, so the output dtype and shape also perform the
	// header-only U8 weight repack and FP8 scale relabeling.
	specs, err := planExpertGroup(group, weights, "", defaultQuantPolicy{})
	if err != nil {
		return nil, err
	}
	bank := &specs[0]
	weightsByProjection := bank.Tensors
	bank.Tensors = nil
	for _, packed := range weightsByProjection {
		count := len(packed.Sources)
		scaleSources := make([]SourceTensor, count)
		globalSources := make([]SourceTensor, count)
		for index, weight := range packed.Sources {
			part, ok := fused[weight.Name]
			if !ok || part.Metadata["quant_type"] != "nvfp4" || len(part.Tensors) != 3 ||
				len(weight.Shape) != 2 {
				return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 layout", weight.Name)
			}
			wordShape := weight.Shape
			normalized := part.Tensors[0]
			switch weight.Dtype {
			case "U32":
				if normalized.Transform != TransformNone {
					return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 weight transform", weight.Name)
				}
			case "U8":
				if normalized.Transform != TransformRepackFP4 || normalized.OutDtype != "U32" ||
					len(normalized.OutShape) != 2 || weight.Shape[1]%4 != 0 {
					return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 weight transform", weight.Name)
				}
				wordShape = normalized.OutShape
			default:
				return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 weight dtype %s", weight.Name, weight.Dtype)
			}
			if index == 0 {
				packed.OutDtype = "U32"
				packed.OutShape = append([]int32{int32(count)}, wordShape...)
			} else if !slices.Equal(wordShape, packed.OutShape[1:]) {
				return nil, fmt.Errorf("prequantized expert %s differs from the NVFP4 bank layout", weight.Name)
			}
			scale, global := part.Tensors[1], part.Tensors[2]
			if scale.Name != weight.Name+".scale" || global.Name != weight.Name+".global_scale" ||
				len(scale.Sources) != 1 || len(global.Sources) != 1 ||
				global.Transform != TransformScalarF32 {
				return nil, fmt.Errorf("prequantized expert %s has unsupported NVFP4 companions", weight.Name)
			}
			s, g := scale.Sources[0], global.Sources[0]
			scaleIsU8 := s.Dtype == "U8" && scale.Transform == TransformNone
			scaleIsFP8 := isE4M3Dtype(s.Dtype) && scale.Transform == TransformRelabelU8 && scale.OutDtype == "U8"
			if (!scaleIsU8 && !scaleIsFP8) || len(s.Shape) != 2 || g.Dtype != "F32" || !slices.Equal(g.Shape, []int32{1}) ||
				(index > 0 && (s.Dtype != scaleSources[0].Dtype || !slices.Equal(s.Shape, scaleSources[0].Shape) || !slices.Equal(g.Shape, globalSources[0].Shape))) {
				return nil, fmt.Errorf("prequantized expert %s differs from the NVFP4 bank layout", weight.Name)
			}
			if bank.Metadata != nil && !maps.Equal(bank.Metadata, part.Metadata) {
				return nil, fmt.Errorf("prequantized expert %s has different quantization metadata", weight.Name)
			}
			bank.Metadata = part.Metadata
			scaleSources[index], globalSources[index] = s, g
		}
		bank.Tensors = append(bank.Tensors, packed,
			TensorSpec{Name: packed.Name + ".scale", Sources: scaleSources, Transform: TransformStackExperts, OutDtype: "U8", OutShape: append([]int32{int32(count)}, scaleSources[0].Shape...)},
			TensorSpec{Name: packed.Name + ".global_scale", Sources: globalSources, Transform: TransformStackExperts, OutDtype: "F32", OutShape: []int32{int32(count)}},
		)
	}
	return specs, nil
}

// findPrequant returns the prequantized producer pattern a weight tensor
// matches and the weight's base name. It returns ok=false when name is not a
// prequantized weight (a companion or a plain tensor).
func findPrequant(name string, inv Inventory) (prequantPattern, string, bool) {
	for _, p := range prequantPatterns {
		base, ok := strings.CutSuffix(name, p.weightSuffix)
		if !ok {
			continue
		}
		scaleSrc := base + p.scaleSuffix
		if !inv.Has(scaleSrc) {
			continue
		}
		if matchesDtype(inv.Tensors[name].Dtype, p.weightDtypes) && matchesDtype(inv.Tensors[scaleSrc].Dtype, p.scaleDtypes) {
			return p, base, true
		}
	}
	return prequantPattern{}, "", false
}

// matchPrequant returns the fused blob for a weight tensor if it matches a
// prequantized producer, along with the source names it consumes. It returns
// ok=false when name is not a prequantized weight (a companion or a plain
// tensor).
func matchPrequant(name string, inv Inventory) (BlobSpec, []string, bool) {
	p, base, ok := findPrequant(name, inv)
	if !ok {
		return BlobSpec{}, nil, false
	}
	scaleSrc := base + p.scaleSuffix
	weight := inv.Tensors[name]
	scale := inv.Tensors[scaleSrc]

	outWeight := base + ".weight"
	var tensors []TensorSpec
	var consumed []string

	weightTensor := TensorSpec{Name: outWeight, Sources: []SourceTensor{weight}}
	if p.repackWeight && len(weight.Shape) == 2 && (strings.EqualFold(weight.Dtype, "U8") || isE4M3Dtype(weight.Dtype)) {
		weightTensor.Transform = TransformRepackFP4
		if isE4M3Dtype(weight.Dtype) {
			weightTensor.Transform = TransformRepackFP8
		}
		weightTensor.OutDtype = "U32"
		weightTensor.OutShape = []int32{weight.Shape[0], weight.Shape[1] / 4}
	}
	tensors = append(tensors, weightTensor)

	scaleTensor := TensorSpec{Name: outWeight + ".scale", Sources: []SourceTensor{scale}}
	if p.scaleRelabelU8 && isE4M3Dtype(scale.Dtype) {
		scaleTensor.Transform = TransformRelabelU8
		scaleTensor.OutDtype = "U8"
	}
	tensors = append(tensors, scaleTensor)
	consumed = append(consumed, scaleSrc)

	if p.biasSuffix != "" {
		if biasSrc := base + p.biasSuffix; inv.Has(biasSrc) {
			tensors = append(tensors, TensorSpec{Name: outWeight + ".bias", Sources: []SourceTensor{inv.Tensors[biasSrc]}})
			consumed = append(consumed, biasSrc)
		}
	}

	if p.globalSuffix != "" {
		if gSrc := base + p.globalSuffix; inv.Has(gSrc) {
			globalSource := inv.Tensors[gSrc]
			global := TensorSpec{Name: outWeight + ".global_scale", Sources: []SourceTensor{globalSource}, Transform: f32ScaleTransform(globalSource)}
			if p.globalReciprocal {
				global.Transform = TransformReciprocalF32
			}
			tensors = append(tensors, global)
			consumed = append(consumed, gSrc)
		}
	}

	for _, suffix := range p.ignoreSuffixes {
		if source := base + suffix; inv.Has(source) {
			consumed = append(consumed, source)
		}
	}

	md := prequantMetadata(inv, p)
	// checkModuleTypes has refused entries that resolve to no type.
	if m, ok := moduleEntry(inv, p, base); ok {
		quantType, groupSize := m.settings()
		md = map[string]string{"quant_type": quantType, "group_size": strconv.Itoa(groupSize)}
	}
	return BlobSpec{Name: outWeight, Tensors: tensors, Metadata: md}, consumed, true
}

// checkModuleEntry rejects a per-module entry whose settings can't be read or
// name a type the engine lacks.
func checkModuleEntry(path string, m moduleQuantization) error {
	if m.Unreadable {
		return fmt.Errorf("%s: config.json's quantization entry for this module has unreadable bits, group_size or mode", path)
	}
	if quantType, _ := m.settings(); quantType == "" {
		// Only an unknown mode, or an affine width the engine lacks, resolves
		// to no type.
		setting := fmt.Sprintf("mode %q", m.Mode)
		if strings.EqualFold(cmp.Or(m.Mode, "affine"), "affine") {
			setting = fmt.Sprintf("%d bits", m.Bits)
		}
		return fmt.Errorf("%s: config.json quantizes this module with %s, which is not supported", path, setting)
	}
	return nil
}

// checkAffineWidth verifies that a fused affine blob's packed weight and scales
// agree with the bit width and group size recorded in its metadata. The
// metadata comes from the weight's per-module entry, else the config's default
// (for Flint's embedding, from mlx_export.embedding_bits), and the loader
// trusts it over the tensor shapes, so a tensor quantized differently without
// the config saying so would otherwise load at the wrong width. Shapes only fix
// the product of bits and group size, so a tensor that changes both inversely
// without an entry passes.
func checkAffineWidth(spec BlobSpec) error {
	groupSize, bits, mode := quant.Params(spec.Metadata["quant_type"])
	if mode != "affine" {
		return nil
	}
	if gs, err := strconv.Atoi(spec.Metadata["group_size"]); err == nil {
		groupSize = gs
	}
	weightIndex := slices.IndexFunc(spec.Tensors, func(t TensorSpec) bool { return t.Name == spec.Name })
	scaleIndex := slices.IndexFunc(spec.Tensors, func(t TensorSpec) bool { return t.Name == spec.Name+".scale" })
	if weightIndex < 0 || scaleIndex < 0 {
		return nil
	}
	weight, scale := spec.Tensors[weightIndex].Sources[0], spec.Tensors[scaleIndex].Sources[0]
	if !strings.EqualFold(weight.Dtype, "U32") || len(weight.Shape) == 0 || len(scale.Shape) == 0 {
		return nil
	}
	packed := int64(weight.Shape[len(weight.Shape)-1])
	groups := int64(scale.Shape[len(scale.Shape)-1])
	if packed*32 != groups*int64(groupSize)*int64(bits) {
		return fmt.Errorf("%s: packed width %d with %d scale groups does not match %s with group size %d from config.json", spec.Name, packed, groups, spec.Metadata["quant_type"], groupSize)
	}
	return nil
}

// checkExpertWidth rejects routed MoE experts at an affine width the
// architecture's MLX runtime cannot keep quantized. The MoE runtimes that gate
// on supportsGatherQMM keep routed experts quantized only at 4- and 8-bit
// affine; they dequantize other widths at load (qwen4_exp refuses them),
// multiplying expert memory several times over. Flint's runtime refuses them
// too. Gemma 4 keeps experts quantized at every width.
func checkExpertWidth(spec BlobSpec, arch string) error {
	_, bits, mode := quant.Params(spec.Metadata["quant_type"])
	if mode != "affine" || bits == 4 || bits == 8 || !isRoutedExpertWeight(spec.Name) ||
		strings.Contains(strings.ToLower(arch), "gemma4") {
		return nil
	}
	return fmt.Errorf("%s: %s mixture-of-experts weights are not supported yet; experts must be 4- or 8-bit", spec.Name, spec.Metadata["quant_type"])
}

func f32ScaleTransform(t SourceTensor) Transform {
	for _, dim := range t.Shape {
		if dim != 1 {
			return TransformF32
		}
	}
	return TransformScalarF32
}

func matchesDtype(dtype string, allowed []string) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, want := range allowed {
		if strings.EqualFold(dtype, want) {
			return true
		}
	}
	return false
}

// moduleEntries returns config.json's mlx-lm per-module entries, keyed by
// module path. They apply only when the default quantization names an MLX
// type.
func (cfg sourceModelConfig) moduleEntries() map[string]moduleQuantization {
	q := cfg.declaredQuantization()
	if q.quantType() == "" {
		return nil
	}
	return q.Modules
}

// moduleEntry returns the per-module entry for the fused weight at base, if its
// producer honors entries. mlx-lm names each entry for the module whose weight
// is base+".weight".
func moduleEntry(inv Inventory, p prequantPattern, base string) (moduleQuantization, bool) {
	if !p.moduleEntries {
		return moduleQuantization{}, false
	}
	m, ok := inv.Config.moduleEntries()[base]
	return m, ok
}

// checkModuleTypes rejects per-module entries at a type the engine cannot
// load, or whose settings can't be read. It checks only the entries that apply
// to a weight; matchPrequant records them.
func checkModuleTypes(inv Inventory) error {
	for _, name := range sortedTensorNames(inv) {
		p, base, ok := findPrequant(name, inv)
		if !ok {
			continue
		}
		if m, ok := moduleEntry(inv, p, base); ok {
			if err := checkModuleEntry(base, m); err != nil {
				return err
			}
		}
	}
	return nil
}

// prequantMetadata builds the fused blob's metadata: the source config's quant
// metadata, with the pattern's quant_type override and group_size default
// applied. Returns nil when there is nothing to record.
func prequantMetadata(inv Inventory, p prequantPattern) map[string]string {
	md := make(map[string]string)
	for k, v := range inv.Config.QuantMetadata() {
		md[k] = v
	}
	if p.forceQuantType != "" {
		md["quant_type"] = p.forceQuantType
	}
	if p.defaultGroupSize != "" {
		if _, ok := md["group_size"]; !ok {
			md["group_size"] = p.defaultGroupSize
		}
	}
	if len(md) == 0 {
		return nil
	}
	return md
}
