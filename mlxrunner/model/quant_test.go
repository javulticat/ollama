package model

import "testing"

func TestCheckCUDAQuant(t *testing.T) {
	supported := map[string]*TensorQuantInfo{
		"embed.weight":  {QuantType: "INT2", GroupSize: 128},
		"layer.weight":  {QuantType: "INT4", GroupSize: 64},
		"expert.weight": {QuantType: "NVFP4", GroupSize: 16},
	}
	if err := checkCUDAQuant(supported); err != nil {
		t.Fatalf("checkCUDAQuant() error = %v", err)
	}

	supported["head.weight"] = &TensorQuantInfo{QuantType: "INT1", GroupSize: 128}
	err := checkCUDAQuant(supported)
	if want := "head.weight: int1 weights are not supported on CUDA"; err == nil || err.Error() != want {
		t.Fatalf("checkCUDAQuant() error = %v, want %q", err, want)
	}
}
