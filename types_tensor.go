package ort

import (
	"reflect"
)

type FloatData interface {
	~float32 | ~float64
}

type IntData interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

type ComplexData interface {
	~complex64 | ~complex128
}

// This is used as a type constraint for the generic Tensor type.
type TensorData interface {
	FloatData | IntData | ComplexData | string | bool
}

type TensorElementDataType int

const (
	TensorElementDataType_Undefined = TensorElementDataType(iota) // TensorElementDataType[float32]
	TensorElementDataType_FLOAT                                   // TensorElementDataType[float32]
	TensorElementDataType_UINT8                                   // TensorElementDataType[uint8]
	TensorElementDataType_INT8                                    // TensorElementDataType[int8]
	TensorElementDataType_UINT16                                  // TensorElementDataType[uint16]
	TensorElementDataType_INT16                                   // TensorElementDataType[int16]
	TensorElementDataType_INT32                                   // TensorElementDataType[int32]
	TensorElementDataType_INT64                                   // TensorElementDataType[int64]
	TensorElementDataType_STRING                                  // TensorElementDataType[string]
	TensorElementDataType_BOOL                                    // TensorElementDataType[bool]
	//// feature = "half"
	/// 16-bit floating point number, but golang no float16, so using float32
	TensorElementDataType_FLOAT16    // TensorElementDataType[float32]
	TensorElementDataType_DOUBLE     // TensorElementDataType[float64]
	TensorElementDataType_UINT32     // TensorElementDataType[uint32]
	TensorElementDataType_UINT64     // TensorElementDataType[uint64]
	TensorElementDataType_COMPLEX64  // TensorElementDataType[complex64]
	TensorElementDataType_COMPLEX128 // TensorElementDataType[complex128]
	/// Brain 16-bit floating point number, equivalent to `half::bf16`
	TensorElementDataType_BFLOAT16 // TensorElementDataType[float32]
)

// Returns the ONNX enum value used to indicate TensorData type T.
func GetOnnxTensorElementDataType[T TensorData]() ONNXTensorElementDataType {
	// Sadly, we can't do type assertions to get underlying types, so we need
	// to use reflect here instead.
	var v T
	kind := reflect.ValueOf(v).Kind()
	switch kind {
	case reflect.Float64:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
	case reflect.Float32:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	case reflect.Int8:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
	case reflect.Uint8:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
	case reflect.Int16:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
	case reflect.Uint16:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
	case reflect.Int32:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
	case reflect.Uint32:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
	case reflect.Int64:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	case reflect.Uint64:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
	}
	return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}
