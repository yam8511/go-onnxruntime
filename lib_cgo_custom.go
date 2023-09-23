//go:build onnxenv
// +build onnxenv

package ort

// need set env CGO_CFLAGS="-O2 -g -I path/to/include"
// need set env CGO_LDFLAGS="-O2 -g -L path/to/lib -lonnxruntime"

// #include "onnxruntime_wrapper.h"
import "C"
import "fmt"

// ########################################################
// Version
// ########################################################

const ORT_API_VERSION = C.ORT_API_VERSION

// ########################################################
// Struct
// ########################################################

type (
	OrtAllocator               = C.OrtAllocator
	OrtKernelInfo              = C.OrtKernelInfo
	OrtKernelContext           = C.OrtKernelContext
	OrtCustomOp                = C.OrtCustomOp
	OrtROCMProviderOptions     = C.OrtROCMProviderOptions
	OrtTensorRTProviderOptions = C.OrtTensorRTProviderOptions
	OrtMIGraphXProviderOptions = C.OrtMIGraphXProviderOptions
	OrtOpenVINOProviderOptions = C.OrtOpenVINOProviderOptions
	OrtApi                     = C.OrtApi
	OrtTrainingApi             = C.OrtTrainingApi
	OrtApiBase                 = C.OrtApiBase
)

/** \addtogroup Global
 * ONNX Runtime C API
 * @{
 */
// The actual types defined have an Ort prefix
type (
	OrtEnv                       = C.OrtEnv
	OrtStatus                    = C.OrtStatus // nullptr for Status* indicates success
	OrtMemoryInfo                = C.OrtMemoryInfo
	OrtIoBinding                 = C.OrtIoBinding
	OrtSession                   = C.OrtSession // Don't call ReleaseSession from Dllmain (because session owns a thread pool)
	OrtValue                     = C.OrtValue
	OrtRunOptions                = C.OrtRunOptions
	OrtTypeInfo                  = C.OrtTypeInfo
	OrtTensorTypeAndShapeInfo    = C.OrtTensorTypeAndShapeInfo
	OrtMapTypeInfo               = C.OrtMapTypeInfo
	OrtSequenceTypeInfo          = C.OrtSequenceTypeInfo
	OrtSessionOptions            = C.OrtSessionOptions
	OrtCustomOpDomain            = C.OrtCustomOpDomain
	OrtModelMetadata             = C.OrtModelMetadata
	OrtThreadingOptions          = C.OrtThreadingOptions
	OrtArenaCfg                  = C.OrtArenaCfg
	OrtPrepackedWeightsContainer = C.OrtPrepackedWeightsContainer
	OrtTensorRTProviderOptionsV2 = C.OrtTensorRTProviderOptionsV2
	OrtCUDAProviderOptionsV2     = C.OrtCUDAProviderOptionsV2
	OrtCANNProviderOptions       = C.OrtCANNProviderOptions
	OrtOp                        = C.OrtOp
	OrtOpAttr                    = C.OrtOpAttr
)

// ########################################################
// Enum
// ########################################################

type ONNXTensorElementDataType = C.ONNXTensorElementDataType

const (
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED  ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT      ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT  // maps to c type float
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8      ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8  // maps to c type uint8_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8       ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8   // maps to c type int8_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16     ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 // maps to c type uint16_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16      ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16  // maps to c type int16_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32      ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32  // maps to c type int32_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64      ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64  // maps to c type int64_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING     ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING // maps to c++ type std::string
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL       ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16    ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
	ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE     ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE     // maps to c type double
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32     ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32     // maps to c type uint32_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64     ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64     // maps to c type uint64_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64  ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64  // complex with float32 real and imaginary components
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 // complex with float64 real and imaginary components
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16   ONNXTensorElementDataType = C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16   // Non-IEEE floating-point format based on IEEE754 single-precision
)

func (t ONNXTensorElementDataType) String() string {
	switch t {
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		return "UNDEFINED"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return "FLOAT"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return "UINT8"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return "INT8"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		return "UINT16"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return "INT16"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return "INT32"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return "INT64"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		return "STRING"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		return "BOOL"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		return "FLOAT16"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return "DOUBLE"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		return "UINT32"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		return "UINT64"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		return "COMPLEX64"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
		return "COMPLEX128"
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		return "BFLOAT16"
	default:
		return fmt.Sprintf("Unknown Type: 0x%x", int(t))
	}
}

type ONNXType C.ONNXType

const (
	ONNX_TYPE_UNKNOWN      ONNXType = C.ONNX_TYPE_UNKNOWN
	ONNX_TYPE_TENSOR       ONNXType = C.ONNX_TYPE_TENSOR
	ONNX_TYPE_SEQUENCE     ONNXType = C.ONNX_TYPE_SEQUENCE
	ONNX_TYPE_MAP          ONNXType = C.ONNX_TYPE_MAP
	ONNX_TYPE_OPAQUE       ONNXType = C.ONNX_TYPE_OPAQUE
	ONNX_TYPE_SPARSETENSOR ONNXType = C.ONNX_TYPE_SPARSETENSOR
	ONNX_TYPE_OPTIONAL     ONNXType = C.ONNX_TYPE_OPTIONAL
)

func (t ONNXType) String() string {
	switch t {
	case ONNX_TYPE_UNKNOWN:
		return "UNKNOWN"
	case ONNX_TYPE_TENSOR:
		return "TENSOR"
	case ONNX_TYPE_SEQUENCE:
		return "SEQUENCE"
	case ONNX_TYPE_MAP:
		return "MAP"
	case ONNX_TYPE_OPAQUE:
		return "OPAQUE"
	case ONNX_TYPE_SPARSETENSOR:
		return "SPARSETENSOR"
	case ONNX_TYPE_OPTIONAL:
		return "OPTIONAL"
	default:
		return fmt.Sprintf("Unknown Type: 0x%x", int(t))
	}
}

type OrtSparseFormat C.OrtSparseFormat

const (
	ORT_SPARSE_UNDEFINED    OrtSparseFormat = C.ORT_SPARSE_UNDEFINED
	ORT_SPARSE_COO          OrtSparseFormat = C.ORT_SPARSE_COO
	ORT_SPARSE_CSRC         OrtSparseFormat = C.ORT_SPARSE_CSRC
	ORT_SPARSE_BLOCK_SPARSE OrtSparseFormat = C.ORT_SPARSE_BLOCK_SPARSE
)

type OrtLoggingLevel C.OrtLoggingLevel

const (
	ORT_LOGGING_LEVEL_VERBOSE OrtLoggingLevel = C.ORT_LOGGING_LEVEL_VERBOSE ///< Verbose informational messages (least severe).
	ORT_LOGGING_LEVEL_INFO    OrtLoggingLevel = C.ORT_LOGGING_LEVEL_INFO    ///< Informational messages.
	ORT_LOGGING_LEVEL_WARNING OrtLoggingLevel = C.ORT_LOGGING_LEVEL_WARNING ///< Warning messages.
	ORT_LOGGING_LEVEL_ERROR   OrtLoggingLevel = C.ORT_LOGGING_LEVEL_ERROR   ///< Error messages.
	ORT_LOGGING_LEVEL_FATAL   OrtLoggingLevel = C.ORT_LOGGING_LEVEL_FATAL   ///< Fatal error messages (most severe).
)

type OrtErrorCode C.OrtErrorCode

const (
	ORT_OK                OrtErrorCode = C.ORT_OK
	ORT_FAIL              OrtErrorCode = C.ORT_FAIL
	ORT_INVALID_ARGUMENT  OrtErrorCode = C.ORT_INVALID_ARGUMENT
	ORT_NO_SUCHFILE       OrtErrorCode = C.ORT_NO_SUCHFILE
	ORT_NO_MODEL          OrtErrorCode = C.ORT_NO_MODEL
	ORT_ENGINE_ERROR      OrtErrorCode = C.ORT_ENGINE_ERROR
	ORT_RUNTIME_EXCEPTION OrtErrorCode = C.ORT_RUNTIME_EXCEPTION
	ORT_INVALID_PROTOBUF  OrtErrorCode = C.ORT_INVALID_PROTOBUF
	ORT_MODEL_LOADED      OrtErrorCode = C.ORT_MODEL_LOADED
	ORT_NOT_IMPLEMENTED   OrtErrorCode = C.ORT_NOT_IMPLEMENTED
	ORT_INVALID_GRAPH     OrtErrorCode = C.ORT_INVALID_GRAPH
	ORT_EP_FAIL           OrtErrorCode = C.ORT_EP_FAIL
)

type OrtOpAttrType C.OrtOpAttrType

const (
	ORT_OP_ATTR_UNDEFINED OrtOpAttrType = C.ORT_OP_ATTR_UNDEFINED
	ORT_OP_ATTR_INT       OrtOpAttrType = C.ORT_OP_ATTR_INT
	ORT_OP_ATTR_INTS      OrtOpAttrType = C.ORT_OP_ATTR_INTS
	ORT_OP_ATTR_FLOAT     OrtOpAttrType = C.ORT_OP_ATTR_FLOAT
	ORT_OP_ATTR_FLOATS    OrtOpAttrType = C.ORT_OP_ATTR_FLOATS
	ORT_OP_ATTR_STRING    OrtOpAttrType = C.ORT_OP_ATTR_STRING
	ORT_OP_ATTR_STRINGS   OrtOpAttrType = C.ORT_OP_ATTR_STRINGS
)

type GraphOptimizationLevel C.GraphOptimizationLevel

const (
	ORT_DISABLE_ALL     GraphOptimizationLevel = C.ORT_DISABLE_ALL
	ORT_ENABLE_BASIC    GraphOptimizationLevel = C.ORT_ENABLE_BASIC
	ORT_ENABLE_EXTENDED GraphOptimizationLevel = C.ORT_ENABLE_EXTENDED
	ORT_ENABLE_ALL      GraphOptimizationLevel = C.ORT_ENABLE_ALL
)

type ExecutionMode C.ExecutionMode

const (
	ORT_SEQUENTIAL ExecutionMode = C.ORT_SEQUENTIAL
	ORT_PARALLEL   ExecutionMode = C.ORT_PARALLEL
)

type OrtLanguageProjection C.OrtLanguageProjection

const (
	ORT_PROJECTION_C         OrtLanguageProjection = C.ORT_PROJECTION_C
	ORT_PROJECTION_CPLUSPLUS OrtLanguageProjection = C.ORT_PROJECTION_CPLUSPLUS
	ORT_PROJECTION_CSHARP    OrtLanguageProjection = C.ORT_PROJECTION_CSHARP
	ORT_PROJECTION_PYTHON    OrtLanguageProjection = C.ORT_PROJECTION_PYTHON
	ORT_PROJECTION_JAVA      OrtLanguageProjection = C.ORT_PROJECTION_JAVA
	ORT_PROJECTION_WINML     OrtLanguageProjection = C.ORT_PROJECTION_WINML
	ORT_PROJECTION_NODEJS    OrtLanguageProjection = C.ORT_PROJECTION_NODEJS
)

type OrtAllocatorType C.OrtAllocatorType

const (
	OrtInvalidAllocator OrtAllocatorType = C.OrtInvalidAllocator
	OrtDeviceAllocator  OrtAllocatorType = C.OrtDeviceAllocator
	OrtArenaAllocator   OrtAllocatorType = C.OrtArenaAllocator
)

type OrtMemType C.OrtMemType

const (
	OrtMemTypeCPUInput  OrtMemType = C.OrtMemTypeCPUInput  ///< Any CPU memory used by non-CPU execution provider
	OrtMemTypeCPUOutput OrtMemType = C.OrtMemTypeCPUOutput ///< CPU accessible memory outputted by non-CPU execution provider i.e. CUDA_PINNED
	OrtMemTypeCPU       OrtMemType = C.OrtMemTypeCPU       ///< Temporary CPU accessible memory allocated by non-CPU execution provider i.e. CUDA_PINNED
	OrtMemTypeDefault   OrtMemType = C.OrtMemTypeDefault   ///< The default allocator for execution provider
)

type OrtMemoryInfoDeviceType C.OrtMemoryInfoDeviceType

const (
	OrtMemoryInfoDeviceType_CPU  OrtMemoryInfoDeviceType = C.OrtMemoryInfoDeviceType_CPU
	OrtMemoryInfoDeviceType_GPU  OrtMemoryInfoDeviceType = C.OrtMemoryInfoDeviceType_GPU
	OrtMemoryInfoDeviceType_FPGA OrtMemoryInfoDeviceType = C.OrtMemoryInfoDeviceType_FPGA
)

// type OrtCudnnConvAlgoSearch C.OrtCudnnConvAlgoSearch

const (
	OrtCudnnConvAlgoSearchExhaustive = C.OrtCudnnConvAlgoSearchExhaustive // expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
	OrtCudnnConvAlgoSearchHeuristic  = C.OrtCudnnConvAlgoSearchHeuristic  // lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
	OrtCudnnConvAlgoSearchDefault    = C.OrtCudnnConvAlgoSearchDefault    // default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
)

type OrtCustomOpInputOutputCharacteristic C.OrtCustomOpInputOutputCharacteristic

const (
	INPUT_OUTPUT_REQUIRED OrtCustomOpInputOutputCharacteristic = C.INPUT_OUTPUT_REQUIRED
	INPUT_OUTPUT_OPTIONAL OrtCustomOpInputOutputCharacteristic = C.INPUT_OUTPUT_OPTIONAL
	INPUT_OUTPUT_VARIADIC OrtCustomOpInputOutputCharacteristic = C.INPUT_OUTPUT_VARIADIC
)
