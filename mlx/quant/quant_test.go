package quant

import "testing"

func TestParams(t *testing.T) {
	tests := []struct {
		in        string
		groupSize int
		bits      int
		mode      string
	}{
		{"nvfp4", 16, 4, "nvfp4"},
		{"NVFP4", 16, 4, "nvfp4"},
		{"mxfp4", 32, 4, "mxfp4"},
		{"mxfp8", 32, 8, "mxfp8"},
		{"int4", 64, 4, "affine"},
		{"int8", 64, 8, "affine"},
		{"int1", 64, 1, "affine"},
		{"INT2", 64, 2, "affine"},
		{"int3", 64, 3, "affine"},
		{"int5", 64, 5, "affine"},
		{"int6", 64, 6, "affine"},
		{"fp4", 64, 4, "affine"},
		{"q4", 64, 4, "affine"},
		{"fp8", 64, 8, "affine"},
		{"q8", 64, 8, "affine"},
		{"", 0, 0, ""},
		// Unknown non-empty types fall back to 8-bit affine, matching the
		// runtime loader's historical behavior.
		{"something-else", 32, 8, "affine"},
	}
	for _, tt := range tests {
		gs, bits, mode := Params(tt.in)
		if gs != tt.groupSize || bits != tt.bits || mode != tt.mode {
			t.Errorf("Params(%q) = (%d, %d, %q), want (%d, %d, %q)", tt.in, gs, bits, mode, tt.groupSize, tt.bits, tt.mode)
		}
	}
}

func TestCanonicalAndRequestable(t *testing.T) {
	tests := []struct {
		in          string
		canonical   string
		requestable bool
	}{
		{"int4", "int4", true},
		{"Q8", "int8", true},
		{"nvfp4", "nvfp4", true},
		{"MXFP8", "mxfp8", true},
		{"int1", "int1", false},
		{" INT2 ", "int2", false},
		{"int3", "int3", false},
		{"int5", "int5", false},
		{"int6", "int6", false},
		{"int7", "", false}, // MLX has no 7-bit affine mode
		{"int16", "", false},
		{"q2", "", false},
		{"", "", false},
	}
	for _, tt := range tests {
		if got := Canonical(tt.in); got != tt.canonical {
			t.Errorf("Canonical(%q) = %q, want %q", tt.in, got, tt.canonical)
		}
		if got := Requestable(tt.in); got != tt.requestable {
			t.Errorf("Requestable(%q) = %v, want %v", tt.in, got, tt.requestable)
		}
	}
}

func TestBitsAndUnpackedWidth(t *testing.T) {
	tests := []struct {
		in       string
		bits     int
		packed   int64
		unpacked int64
	}{
		{"int4", 4, 320, 2560},
		{"nvfp4", 4, 320, 2560},
		{"mxfp4", 4, 320, 2560}, // regression: mxfp4 was missing from the old show.go switch
		{"int8", 8, 640, 2560},
		{"mxfp8", 8, 640, 2560},
		{"FP8", 8, 640, 2560},
		{"int1", 1, 80, 2560},
		{"int2", 2, 160, 2560},
		{"int3", 3, 240, 2560}, // 3 bits do not divide a word evenly
		{"int5", 5, 400, 2560},
		{"int6", 6, 480, 2560},
		{"", 0, 320, 0},
		{"unknown", 0, 320, 0}, // strict: no fallback for sizing/display
	}
	for _, tt := range tests {
		if got := Bits(tt.in); got != tt.bits {
			t.Errorf("Bits(%q) = %d, want %d", tt.in, got, tt.bits)
		}
		if got := UnpackedWidth(tt.in, tt.packed); got != tt.unpacked {
			t.Errorf("UnpackedWidth(%q, %d) = %d, want %d", tt.in, tt.packed, got, tt.unpacked)
		}
	}
}
