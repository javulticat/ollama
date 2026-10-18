package nn

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// testRotation spans two blocks so per-block application is exercised.
func testRotation(width, block int) *Hadamard {
	signs := make([]float32, width)
	for i := range signs {
		signs[i] = 1
		if (i*5)%7 < 3 {
			signs[i] = -1
		}
	}
	return &Hadamard{Signs: mlx.FromValues(signs, width), Block: block}
}

func testValues(rows, cols int, step int) *mlx.Array {
	values := make([]float32, rows*cols)
	for i := range values {
		values[i] = float32((i*step)%17-8) / 8
	}
	return mlx.FromValues(values, rows, cols)
}

func requireAllClose(t *mlxtest.T, label string, got, want *mlx.Array, tol float64) {
	t.Helper()
	got, want = got.AsType(mlx.DTypeFloat32), want.AsType(mlx.DTypeFloat32)
	mlx.Eval(got, want)
	gotValues, wantValues := got.Floats(), want.Floats()
	if len(gotValues) != len(wantValues) {
		t.Fatalf("%s length = %d, want %d", label, len(gotValues), len(wantValues))
	}
	for i := range wantValues {
		if diff := math.Abs(float64(gotValues[i] - wantValues[i])); !(diff <= tol) {
			t.Fatalf("%s[%d] = %v, want %v (differs by %g)", label, i, gotValues[i], wantValues[i], diff)
		}
	}
}

// A weight stored with the rotation folded into its input dimension (each row
// rotated forward) reproduces the original layer once the input is rotated.
func TestHadamardLinearUnfoldsRotatedWeight(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const width, block = 64, 32
		rotation := testRotation(width, block)
		weight := testValues(5, width, 3)
		x := testValues(3, width, 7)

		layer := &HadamardLinear{Inner: NewLinear(rotation.Forward(weight), nil), Rotation: rotation}
		if got := layer.OutputDim(); got != 5 {
			t.Fatalf("OutputDim() = %d, want 5", got)
		}
		requireAllClose(t, "output", layer.Forward(x), mlx.Matmul(x, mlx.Transpose(weight, 1, 0)), 1e-4)
	})
}

// An embedding stored in the rotated basis looks up the original rows, and its
// tied head scores against the original table.
func TestHadamardEmbeddingUnfoldsRotatedTable(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const vocab, width, block = 6, 64, 32
		rotation := testRotation(width, block)
		table := testValues(vocab, width, 5)
		emb := &HadamardEmbedding{Inner: NewEmbedding(rotation.Forward(table)), Rotation: rotation}

		ids := mlx.FromValues([]int32{2, 0, 5}, 3)
		requireAllClose(t, "lookup", emb.Forward(ids), table.TakeAxis(ids, 0), 1e-5)

		hidden := testValues(2, width, 11)
		requireAllClose(t, "tied head", emb.AsLinear().Forward(hidden), mlx.Matmul(hidden, mlx.Transpose(table, 1, 0)), 1e-4)
	})
}

// A quantized inner layer sees the rotated input in the caller's dtype, and
// the output keeps that dtype.
func TestHadamardLinearQuantizedInner(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const width, block, groupSize, bits = 256, 128, 64, 2
		rotation := testRotation(width, block)
		inner := NewQuantizedLinear(testValues(8, width, 3).AsType(mlx.DTypeFloat16), nil, groupSize, bits, "affine")
		x := testValues(2, width, 7).AsType(mlx.DTypeFloat16)

		got := (&HadamardLinear{Inner: inner, Rotation: rotation}).Forward(x)
		if got.DType() != mlx.DTypeFloat16 {
			t.Fatalf("output dtype = %v, want %v", got.DType(), mlx.DTypeFloat16)
		}
		dense := mlx.Dequantize(inner.Weight, inner.Scales, inner.QBiases, groupSize, bits, "affine", nil)
		want := mlx.Matmul(rotation.Forward(x).AsType(mlx.DTypeFloat32), mlx.Transpose(dense.AsType(mlx.DTypeFloat32), 1, 0))
		requireAllClose(t, "output", got, want, 2e-2)
	})
}

func TestHadamardInverseUndoesForward(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		rotation := testRotation(64, 32)
		x := testValues(4, 64, 3).Reshape(2, 2, 64)
		requireAllClose(t, "round trip", rotation.Inverse(rotation.Forward(x)), x, 1e-5)
	})
}
