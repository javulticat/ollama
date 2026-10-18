package nn

import (
	"math"

	"github.com/ollama/ollama/mlx"
)

// Hadamard is a blockwise randomized Hadamard rotation of the last axis:
// features are multiplied by ±1 Signs, then each run of Block features by the
// normalized Walsh-Hadamard matrix. Checkpoints that fold this rotation into
// their weights (Prism ML's Bonsai 2) need it applied to activations at
// runtime. The rotation is computed in float32 and cast back.
type Hadamard struct {
	Signs *mlx.Array
	Block int
}

// Forward rotates x into the basis of weights stored with the rotation folded
// into their input dimension.
func (h *Hadamard) Forward(x *mlx.Array) *mlx.Array {
	return h.transform(mlx.Mul(x.AsType(mlx.DTypeFloat32), h.Signs)).AsType(x.DType())
}

// Inverse undoes Forward, mapping rotated values back to the model's basis.
func (h *Hadamard) Inverse(x *mlx.Array) *mlx.Array {
	return mlx.Mul(h.transform(x.AsType(mlx.DTypeFloat32)), h.Signs).AsType(x.DType())
}

func (h *Hadamard) transform(x *mlx.Array) *mlx.Array {
	blocks := x.Reshape(-1, h.Block)
	return mlx.HadamardTransform(blocks, float32(1/math.Sqrt(float64(h.Block)))).Reshape(x.Dims()...)
}

// HadamardLinear is a linear layer whose weights were stored with a Hadamard
// rotation folded into their input dimension, so it rotates its input first.
type HadamardLinear struct {
	Inner    LinearLayer
	Rotation *Hadamard
}

func (l *HadamardLinear) Forward(x *mlx.Array) *mlx.Array {
	return l.Inner.Forward(l.Rotation.Forward(x))
}

func (l *HadamardLinear) OutputDim() int32 {
	return l.Inner.OutputDim()
}

// HadamardEmbedding is an embedding table stored in the rotated basis. Lookups
// rotate rows back to the model's basis, and the tied output head rotates its
// input forward to meet the stored rows.
type HadamardEmbedding struct {
	Inner    EmbeddingLayer
	Rotation *Hadamard
}

func (e *HadamardEmbedding) Forward(indices *mlx.Array) *mlx.Array {
	return e.Rotation.Inverse(e.Inner.Forward(indices))
}

func (e *HadamardEmbedding) AsLinear() LinearLayer {
	return &HadamardLinear{Inner: e.Inner.AsLinear(), Rotation: e.Rotation}
}
