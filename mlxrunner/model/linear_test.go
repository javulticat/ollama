package model

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Every MLX affine width loads from its per-tensor metadata, even where the
// packed shapes alone are ambiguous (2-bit/128 packs like 4-bit/64), and its
// matmul agrees with the dequantized weight.
func TestMakeLinearLayerAffineWidths(t *testing.T) {
	const rows, cols, groupSize = 8, 256, 128
	for _, bits := range []int{1, 2, 3, 4, 5, 6, 8} {
		mlxtest.RunSubtest(t, fmt.Sprintf("int%d", bits), func(t *mlxtest.T) {
			weightValues := make([]float32, rows*cols)
			for i := range weightValues {
				weightValues[i] = float32((i*7)%13-6) / 16
			}
			weight := mlx.FromValues(weightValues, rows, cols).AsType(mlx.DTypeFloat16)
			packed, scales, qbiases := mlx.Quantize(weight, groupSize, bits, "affine")
			mlx.Eval(packed, scales, qbiases)

			tensors := map[string]*mlx.Array{
				"l.weight":       packed,
				"l.weight_scale": scales,
				"l.weight_qbias": qbiases,
			}
			tensorQuant := map[string]*TensorQuantInfo{
				"l.weight": {QuantType: fmt.Sprintf("INT%d", bits), GroupSize: groupSize},
			}
			// The model-wide defaults deliberately disagree with the tensor.
			layer := MakeLinearLayer(tensors, "l", 64, 4, "affine", tensorQuant)
			ql, ok := layer.(*nn.QuantizedLinear)
			if !ok {
				t.Fatalf("layer type = %T, want *nn.QuantizedLinear", layer)
			}
			if ql.GroupSize != groupSize || ql.Bits != bits || ql.Mode != "affine" {
				t.Fatalf("quant params = (%d, %d, %q), want (%d, %d, %q)", ql.GroupSize, ql.Bits, ql.Mode, groupSize, bits, "affine")
			}

			inputValues := make([]float32, 2*cols)
			for i := range inputValues {
				inputValues[i] = float32(i%9-4) / 8
			}
			x := mlx.FromValues(inputValues, 2, cols).AsType(mlx.DTypeFloat16)

			got := ql.Forward(x).AsType(mlx.DTypeFloat32)
			dense := mlx.Dequantize(packed, scales, qbiases, groupSize, bits, "affine", nil).AsType(mlx.DTypeFloat32)
			want := mlx.Matmul(x.AsType(mlx.DTypeFloat32), mlx.Transpose(dense, 1, 0))
			mlx.Eval(got, want)

			gotValues, wantValues := got.Floats(), want.Floats()
			for i := range wantValues {
				tolerance := 0.01 * math.Max(math.Abs(float64(wantValues[i])), 1)
				if delta := math.Abs(float64(gotValues[i] - wantValues[i])); !(delta <= tolerance) {
					t.Fatalf("output[%d] = %v, want %v (delta %v > %v)", i, gotValues[i], wantValues[i], delta, tolerance)
				}
			}
		})
	}
}
