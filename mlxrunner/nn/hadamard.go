package nn

import (
	"fmt"
	"math"
	"sync"

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
	return h.kernels().forward(x, h.Signs)
}

// Inverse undoes Forward, mapping rotated values back to the model's basis.
func (h *Hadamard) Inverse(x *mlx.Array) *mlx.Array {
	return h.kernels().inverse(x, h.Signs)
}

func (h *Hadamard) kernels() *hadamardKernels {
	return hadamardKernelsFor(h.Block, h.Signs.Dim(0))
}

// hadamardKernels are the rotations for one block size and feature width,
// compiled so the casts and sign multiply around the transform fuse into
// fewer kernels.
type hadamardKernels struct {
	forward, inverse func(x, signs *mlx.Array) *mlx.Array
}

type hadamardShape struct{ block, width int }

var (
	hadamardKernelsMu      sync.Mutex
	hadamardKernelsByShape = make(map[hadamardShape]*hadamardKernels)
)

// hadamardKernelsFor compiles a shape's rotations on first use. Each compiled
// graph is shapeless only in the leading dimensions: the feature width is
// fixed per graph, since a shapeless trace bakes in the block count.
func hadamardKernelsFor(block, width int) *hadamardKernels {
	hadamardKernelsMu.Lock()
	defer hadamardKernelsMu.Unlock()
	shape := hadamardShape{block, width}
	if k := hadamardKernelsByShape[shape]; k != nil {
		return k
	}
	scale := float32(1 / math.Sqrt(float64(block)))
	// Unflatten and Flatten, unlike reshapes to the traced dimensions, keep
	// the compiled graph valid as the leading dimensions change.
	transform := func(x *mlx.Array) *mlx.Array {
		return mlx.HadamardTransform(x.Unflatten(-1, width/block, block), scale).Flatten(-2, -1)
	}
	k := &hadamardKernels{
		forward: mlx.Compile2(fmt.Sprintf("HadamardForward%dx%d", width, block), func(x, signs *mlx.Array) *mlx.Array {
			return transform(mlx.Mul(x.AsType(mlx.DTypeFloat32), signs)).AsType(x.DType())
		}, mlx.Shapeless()),
		inverse: mlx.Compile2(fmt.Sprintf("HadamardInverse%dx%d", width, block), func(x, signs *mlx.Array) *mlx.Array {
			return mlx.Mul(transform(x.AsType(mlx.DTypeFloat32)), signs).AsType(x.DType())
		}, mlx.Shapeless()),
	}
	hadamardKernelsByShape[shape] = k
	return k
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
