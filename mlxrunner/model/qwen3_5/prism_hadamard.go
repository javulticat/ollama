package qwen3_5

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Prism ML's Hadamard packs (model_type prism_hadamard_qwen35, the Ternary
// Bonsai 2 family) are Qwen 3.5 checkpoints whose language-model projections
// were rotated by a blockwise randomized Hadamard transform before
// quantization, folding the rotation into each weight's input dimension. The
// config's modules list names every packed module with its rotation block, and
// each rotated module ships its ±1 signs as <module>.signs. Loaded without the
// matching activation rotations the weights decode garbage without any error,
// so the pack is validated strictly, following mlx-vlm's prism_hadamard_qwen35.

func init() {
	model.Register("prism_hadamard_qwen35", newPrismHadamardModel)
}

// hadamardModule is one entry of a Prism pack's packed module manifest. Path
// is relative to the language model, and Block 0 marks a module that is
// quantized but not rotated.
type hadamardModule struct {
	Path      string `json:"path"`
	Block     int    `json:"block"`
	Embedding bool   `json:"embedding"`
	DType     string `json:"dtype"`
}

func newPrismHadamardModel(root *model.Root) (model.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	modules, err := parsePrismHadamardConfig(configData)
	if err != nil {
		return nil, err
	}
	base, err := NewModel(root)
	if err != nil {
		return nil, err
	}
	m := base.(*Model)
	m.hadamardModules = modules
	return m, nil
}

func parsePrismHadamardConfig(configData []byte) ([]hadamardModule, error) {
	var cfg struct {
		SchemaVersion       int              `json:"schema_version"`
		BaseModelType       string           `json:"base_model_type"`
		TensorNamespace     string           `json:"tensor_namespace"`
		GDNActivationLayout string           `json:"gdn_activation_layout"`
		Modules             []hadamardModule `json:"modules"`
	}
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse Hadamard pack config: %w", err)
	}
	switch {
	case cfg.SchemaVersion != 2 || cfg.BaseModelType != "qwen3_5":
		return nil, fmt.Errorf("unsupported Hadamard pack: schema version %d for base model %q, want 2 for qwen3_5", cfg.SchemaVersion, cfg.BaseModelType)
	case cfg.TensorNamespace != "mlx-vlm-qwen3_5":
		return nil, fmt.Errorf("unsupported Hadamard pack tensor namespace %q", cfg.TensorNamespace)
	case cfg.GDNActivationLayout != "grouped":
		return nil, fmt.Errorf("unsupported Hadamard pack GDN activation layout %q", cfg.GDNActivationLayout)
	case len(cfg.Modules) == 0:
		return nil, fmt.Errorf("Hadamard pack is missing its packed module manifest")
	}

	seen := make(map[string]bool, len(cfg.Modules))
	for _, mod := range cfg.Modules {
		switch {
		case seen[mod.Path]:
			return nil, fmt.Errorf("duplicate packed module %s", mod.Path)
		case mod.DType != "float16":
			return nil, fmt.Errorf("packed module %s: unsupported activation dtype %q", mod.Path, mod.DType)
		case !validHadamardBlock(mod.Block):
			return nil, fmt.Errorf("packed module %s: unsupported Hadamard block %d", mod.Path, mod.Block)
		}
		seen[mod.Path] = true
	}
	return cfg.Modules, nil
}

func validHadamardBlock(block int) bool {
	switch block {
	case 0, 512, 1024, 2048, 4096:
		return true
	default:
		return false
	}
}

// hadamardRotations validates the packed modules against the loaded tensors
// and returns each rotated module's input rotation, keyed by tensor path.
func (m *Model) hadamardRotations(tensors map[string]*mlx.Array, layout tensorPathLayout) (map[string]*nn.Hadamard, error) {
	if layout.containerPrefix != "language_model." {
		return nil, fmt.Errorf("Hadamard pack tensors must be under language_model., found %q", layout.containerPrefix)
	}
	cfg := m.Config
	embedPath := layout.modelPrefix + "embed_tokens"
	rotations := make(map[string]*nn.Hadamard)
	// Bonsai 2 packs give every module of an input width the same signs.
	// Deduplicating rotations by value lets projections of the same input
	// rotate it once.
	type knownRotation struct {
		rotation *nn.Hadamard
		signs    []float32
	}
	var known []knownRotation
	for _, mod := range m.hadamardModules {
		path := layout.containerPrefix + mod.Path
		weight, scales, signs := tensors[path+".weight"], tensors[path+".weight_scale"], tensors[path+".signs"]
		switch {
		case weight == nil:
			return nil, fmt.Errorf("packed module %s: missing weight", mod.Path)
		case scales == nil:
			return nil, fmt.Errorf("packed module %s: weight is not quantized", mod.Path)
		case tensors[path+".bias"] != nil:
			return nil, fmt.Errorf("packed module %s: output bias is unsupported", mod.Path)
		case mod.Embedding != (mod.Path == embedPath):
			return nil, fmt.Errorf("packed module %s: embedding flag does not match the module", mod.Path)
		case mod.Block == 0 && signs != nil:
			return nil, fmt.Errorf("packed module %s: unrotated module carries Hadamard signs", mod.Path)
		case mod.Block == 0:
			continue
		case signs == nil:
			return nil, fmt.Errorf("packed module %s: missing Hadamard signs", mod.Path)
		}

		groupSize, _, _ := model.ResolveLinearQuantParams(cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, path+".weight", weight, scales)
		width := scales.Dim(scales.NumDims()-1) * groupSize
		if signs.NumDims() != 1 || signs.Dim(0) != width || width%mod.Block != 0 {
			return nil, fmt.Errorf("packed module %s: Hadamard signs %v do not fit input width %d with block %d", mod.Path, signs.Dims(), width, mod.Block)
		}
		values := signs.AsType(mlx.DTypeFloat32).Floats()
		for _, v := range values {
			if v != 1 && v != -1 {
				return nil, fmt.Errorf("packed module %s: Hadamard signs must be -1 or +1, found %v", mod.Path, v)
			}
		}
		i := slices.IndexFunc(known, func(k knownRotation) bool {
			return k.rotation.Block == mod.Block && slices.Equal(k.signs, values)
		})
		if i < 0 {
			known = append(known, knownRotation{&nn.Hadamard{Signs: signs, Block: mod.Block}, values})
			i = len(known) - 1
		}
		rotations[path] = known[i].rotation
	}

	for name := range tensors {
		if path, ok := strings.CutSuffix(name, ".signs"); ok && strings.HasPrefix(name, layout.containerPrefix) && rotations[path] == nil {
			return nil, fmt.Errorf("Hadamard signs %s belong to no rotated module", name)
		}
	}
	return rotations, nil
}

// castHadamardActivations casts the pack's float32 language-model tensors to
// float16, the activation dtype its module manifest declares. Left in float32,
// the norm weights would promote the whole residual stream and KV cache. A_log
// keeps its precision for the decay it feeds, and signs are applied in float32
// regardless.
func castHadamardActivations(tensors map[string]*mlx.Array, containerPrefix string) {
	for name, t := range tensors {
		if t == nil || t.DType() != mlx.DTypeFloat32 || !strings.HasPrefix(name, containerPrefix) ||
			strings.HasSuffix(strings.ToLower(name), ".a_log") || strings.HasSuffix(name, ".signs") {
			continue
		}
		tensors[name] = t.AsType(mlx.DTypeFloat16)
	}
}

// rotatingLinears builds a Prism pack's layers, wrapping each rotated module
// so its input is rotated into the basis its weights are stored in.
type rotatingLinears struct {
	model.LinearFactory
	rotations map[string]*nn.Hadamard
	applied   map[string]bool
}

func newRotatingLinears(factory model.LinearFactory, rotations map[string]*nn.Hadamard) *rotatingLinears {
	return &rotatingLinears{LinearFactory: factory, rotations: rotations, applied: make(map[string]bool, len(rotations))}
}

func (r *rotatingLinears) Make(path string) nn.LinearLayer {
	layer := r.LinearFactory.Make(path)
	if rotation := r.rotations[path]; rotation != nil && layer != nil {
		r.applied[path] = true
		return &nn.HadamardLinear{Inner: layer, Rotation: rotation}
	}
	return layer
}

func (r *rotatingLinears) embedding(layer nn.EmbeddingLayer, path string) nn.EmbeddingLayer {
	if rotation := r.rotations[path]; rotation != nil {
		r.applied[path] = true
		return &nn.HadamardEmbedding{Inner: layer, Rotation: rotation}
	}
	return layer
}

// shareInputRotations hoists the rotation out of projections that read the
// same input with the same rotation, so a layer rotates that input once
// rather than once per projection: attention's q, k and v, and the MLP's
// gate and up.
func shareInputRotations(layers []*Layer) {
	for _, layer := range layers {
		if attn := layer.FullAttn; attn != nil {
			if rotation, inner, ok := commonRotation(attn.QProj, attn.KProj, attn.VProj); ok {
				attn.InputRotation = rotation
				attn.QProj, attn.KProj, attn.VProj = inner[0], inner[1], inner[2]
			}
		}
		if mlp, ok := layer.MLP.(*DenseMLP); ok {
			if rotation, inner, ok := commonRotation(mlp.GateProj, mlp.UpProj); ok {
				mlp.InputRotation = rotation
				mlp.GateProj, mlp.UpProj = inner[0], inner[1]
			}
		}
	}
}

// commonRotation returns the rotation every layer applies to its input and
// the layers' unrotated projections, or ok=false unless all share one.
func commonRotation(layers ...nn.LinearLayer) (rotation *nn.Hadamard, inner []nn.LinearLayer, ok bool) {
	inner = make([]nn.LinearLayer, len(layers))
	for i, layer := range layers {
		rotated, isRotated := layer.(*nn.HadamardLinear)
		if !isRotated || (rotation != nil && rotated.Rotation != rotation) {
			return nil, nil, false
		}
		rotation, inner[i] = rotated.Rotation, rotated.Inner
	}
	return rotation, inner, true
}

// checkApplied reports a rotated module the model never built, which would
// otherwise have loaded without its rotation.
func (r *rotatingLinears) checkApplied() error {
	for path := range r.rotations {
		if !r.applied[path] {
			return fmt.Errorf("packed module %s is not a projection this model loads", path)
		}
	}
	return nil
}
