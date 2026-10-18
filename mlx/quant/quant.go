// Package quant holds the quantization format facts shared by the model
// importer, the runtime loader, and `ollama show`. It deliberately has no
// dependency on the MLX C library, so any package can use it without pulling
// in cgo — which is what keeps these facts from drifting between separate
// hand-maintained copies.
package quant

import "strings"

type params struct {
	groupSize int
	bits      int
	mode      string
	// requestable marks the types the importer can quantize to. The rest are
	// only recognized in already-quantized sources.
	requestable bool
	// noCUDA marks the types MLX's CUDA backend has no quantized matmul or
	// dequantize kernels for.
	noCUDA bool
}

// byType maps each canonical quantization type to its parameters. Aliases are
// resolved to a canonical name by Canonical before lookup. The affine entries
// cover every bit width MLX supports.
var byType = map[string]params{
	"nvfp4": {groupSize: 16, bits: 4, mode: "nvfp4", requestable: true},
	"mxfp4": {groupSize: 32, bits: 4, mode: "mxfp4", requestable: true},
	"mxfp8": {groupSize: 32, bits: 8, mode: "mxfp8", requestable: true},
	"int1":  {groupSize: 64, bits: 1, mode: "affine", noCUDA: true},
	"int2":  {groupSize: 64, bits: 2, mode: "affine"},
	"int3":  {groupSize: 64, bits: 3, mode: "affine"},
	"int4":  {groupSize: 64, bits: 4, mode: "affine", requestable: true},
	"int5":  {groupSize: 64, bits: 5, mode: "affine"},
	"int6":  {groupSize: 64, bits: 6, mode: "affine"},
	"int8":  {groupSize: 64, bits: 8, mode: "affine", requestable: true},
}

// Canonical returns the canonical name for a quantization type, resolving
// aliases (for example "FP8" and "Q8" both map to "int8"). It returns "" for
// the empty string and for any type it does not recognize.
func Canonical(quantType string) string {
	switch t := strings.ToUpper(strings.TrimSpace(quantType)); t {
	case "NVFP4":
		return "nvfp4"
	case "MXFP4":
		return "mxfp4"
	case "MXFP8":
		return "mxfp8"
	case "FP4", "Q4":
		return "int4"
	case "FP8", "Q8":
		return "int8"
	default:
		// Affine types are named by bit width, for the widths in byType.
		if name := strings.ToLower(t); byType[name].mode == "affine" {
			return name
		}
		return ""
	}
}

// Requestable reports whether the importer can quantize to quantType. Every
// recognized type can be imported prequantized, but only these are offered
// for quantizing a model during create.
func Requestable(quantType string) bool {
	return byType[Canonical(quantType)].requestable
}

// SupportedOnCUDA reports whether MLX's CUDA backend can multiply weights of
// quantType. Unrecognized types report true and are left to the loader.
func SupportedOnCUDA(quantType string) bool {
	return !byType[Canonical(quantType)].noCUDA
}

// Params returns the default group size, bit width, and mode for a
// quantization type. The empty string returns zeros. An unrecognized
// non-empty type falls back to 8-bit affine, matching the runtime loader's
// historical leniency toward unexpected metadata.
func Params(quantType string) (groupSize, bits int, mode string) {
	if strings.TrimSpace(quantType) == "" {
		return 0, 0, ""
	}
	if p, ok := byType[Canonical(quantType)]; ok {
		return p.groupSize, p.bits, p.mode
	}
	return 32, 8, "affine"
}

// Bits returns the bit width of a recognized quantization type, or 0 if the
// type is empty or unrecognized. Unlike Params it applies no fallback, so
// callers that size or display tensors never act on an unknown type.
func Bits(quantType string) int {
	if p, ok := byType[Canonical(quantType)]; ok {
		return p.bits
	}
	return 0
}

// UnpackedWidth returns the logical last dimension of a quantized tensor whose
// stored last dimension is packedWidth, or 0 for an empty or unrecognized
// type. MLX packs quantized values into U32 words as a continuous bit stream,
// so widths such as 3 bits that do not divide 32 still unpack exactly.
func UnpackedWidth(quantType string, packedWidth int64) int64 {
	if b := Bits(quantType); b > 0 {
		return packedWidth * 32 / int64(b)
	}
	return 0
}
