package qwen3_5

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

// A Qwen 3.5 MTP head can also ship as a model of its own, as mlx-vlm's
// qwen3_5_mtp drafters do (the mtp/ folder of a Ternary Bonsai 2 pack).
// Imported with a Modelfile DRAFT, its tensors sit under the manifest's draft
// prefix, and it drafts through the target's embedding and output head, which
// also carry any rotation the target's weights were stored with.

func init() {
	model.RegisterDraft("qwen3_5_mtp", newMTPDraftModel)
}

var _ model.DraftModel = (*mtpDraftModel)(nil)

// mtpDraftModel is an MTP head imported as a separate draft model.
type mtpDraftModel struct {
	Head *MTPHead

	// target is unexported so mlx.Collect does not gather the target's
	// arrays a second time through its draft.
	target       *Model
	tensorPrefix string
}

func newMTPDraftModel(root *model.Root, target model.Model) (model.DraftModel, error) {
	t, ok := target.(*Model)
	if !ok {
		return nil, fmt.Errorf("qwen3_5_mtp draft requires a Qwen 3.5 target, got %T", target)
	}
	configData, err := root.Manifest.ReadConfig(cmp.Or(root.Draft.Config, "draft/config.json"))
	if err != nil {
		return nil, fmt.Errorf("load draft config: %w", err)
	}
	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, fmt.Errorf("parse draft config: %w", err)
	}
	if err := checkMTPDraftConfig(&cfg, t.Config); err != nil {
		return nil, err
	}
	return &mtpDraftModel{target: t, tensorPrefix: cmp.Or(root.Draft.TensorPrefix, "draft.")}, nil
}

// checkMTPDraftConfig verifies that a draft head fits its target: the head's
// layer runs with the target's attention geometry and shares its vocabulary.
func checkMTPDraftConfig(draft, target *Config) error {
	for _, dim := range []struct {
		name          string
		draft, target int32
	}{
		{"hidden_size", draft.HiddenSize, target.HiddenSize},
		{"num_attention_heads", draft.NumAttentionHeads, target.NumAttentionHeads},
		{"num_key_value_heads", draft.NumKeyValueHeads, target.NumKeyValueHeads},
		{"head_dim", draft.HeadDim, target.HeadDim},
		{"num_experts", draft.NumExperts, target.NumExperts},
		{"vocab_size", draft.VocabSize, target.VocabSize},
	} {
		if dim.draft != dim.target {
			return fmt.Errorf("draft %s %d does not match the target's %d", dim.name, dim.draft, dim.target)
		}
	}
	return nil
}

// LoadWeights builds the head from the draft's tensors. mlx-vlm drafters store
// final norm weights; a head that still carries its checkpoint's mtp. prefix
// stores them zero-centered, as mlx-vlm's own loader assumes.
func (d *mtpDraftModel) LoadWeights(tensors map[string]*mlx.Array) error {
	tensorQuant := d.target.TensorQuant
	prefix, shiftNorms := d.tensorPrefix, false
	if fc, _ := tensorByBase(tensors, prefix+"mtp.fc"); fc != nil {
		prefix, shiftNorms = prefix+"mtp.", true
	}

	for name := range tensors {
		if !strings.HasPrefix(name, d.tensorPrefix) {
			continue
		}
		// The draft's config may not count its layers (mlx-vlm writes
		// mtp_num_hidden_layers 0), so the tensors decide.
		if rest, ok := strings.CutPrefix(name, prefix+"layers."); ok {
			if layer, _, _ := strings.Cut(rest, "."); layer != "0" {
				return fmt.Errorf("MTP draft has a layer %s; only a single-layer head is supported", layer)
			}
		}
		// The manifest-wide quantization defaults describe the target, whose
		// width can differ, so each quantized draft tensor needs its own.
		if base, ok := strings.CutSuffix(name, ".weight_scale"); ok && tensorQuant[base+".weight"] == nil {
			return fmt.Errorf("draft tensor %s.weight has no quantization metadata", base)
		}
	}

	linears := model.NewLinearFactory(tensors, 0, 0, "", tensorQuant)
	useQuantizedExperts := anyGatherQMMTensor(tensorQuant, func(name string) bool { return strings.HasPrefix(name, d.tensorPrefix) })
	head, err := loadMTPHead(linears, tensors, d.target.Config, prefix, useQuantizedExperts, shiftNorms)
	if err != nil {
		return fmt.Errorf("draft: %w", err)
	}
	d.Head = head
	return nil
}

func (d *mtpDraftModel) NewCaches() []cache.Cache {
	return []cache.Cache{cache.NewKVCache()}
}

func (d *mtpDraftModel) Forward(b *batch.Batch, _, draftCaches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	return d.target.forwardMTP(d.Head, b, draftCaches[0])
}

func (d *mtpDraftModel) Unembed(x *mlx.Array) *mlx.Array {
	return d.target.Unembed(x)
}
