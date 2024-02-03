package ort

/*
#include "onnxruntime_wrapper.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type OrtSdkOption struct {
	WinDLL_Name  string
	Version      uint32
	LoggingLevel OrtLoggingLevel
}

type OrtSdkArgsF func(opt *OrtSdkOption)

func withOption(args ...OrtSdkArgsF) *OrtSdkOption {
	opts := OrtSdkOption{
		WinDLL_Name:  "onnxruntime.dll",
		Version:      ORT_API_VERSION,
		LoggingLevel: ORT_LOGGING_LEVEL_WARNING,
	}
	for _, f := range args {
		if f != nil {
			f(&opts)
		}
	}
	return &opts
}

func New_ORT_SDK(args ...OrtSdkArgsF) (*ORT_SDK, error) {
	opts := withOption(args...)
	sdk, err := newOrtApi(*opts)
	if err != nil {
		return nil, err
	}

	envName := C.CString("go-ort environment")
	defer C.free(unsafe.Pointer(envName))

	status := C.CreateOrtEnvWithOrtLoggingLevel(
		sdk._Api,
		C.OrtLoggingLevel(opts.LoggingLevel),
		envName, &sdk._Env,
	)
	if status != nil {
		sdk.Release()
		return nil, fmt.Errorf(
			"%w: creating ORT environment: %w",
			ErrExportOrtSdk, sdk.CheckAndReleaseStatus(status),
		)
	}

	{ // 取得預設的Allocator
		sdk._AllocatorPtr, err = sdk.GetAllocatorWithDefaultOptions()
		if err != nil {
			sdk.Release()
			return nil, err
		}
	}

	{ // 建立 MemoryInfo
		sdk._MemoryInfoPrt, err = sdk.CreateCpuMemoryInfo()
		if err != nil {
			sdk.Release()
			return nil, err
		}
	}

	return sdk, nil
}

// Onnxruntime API版本
func (ort *ORT_SDK) ORT_API_VERSION() uint32 { return ort._version }

// Onnxruntime 版本字串
//
// Returns a null terminated string of the version of the Onnxruntime library (eg: "1.8.1")
func (ort *ORT_SDK) GetVersionString() string {
	version := C.GetVersionString(ort._ApiBase)
	return C.GoString(version)
}

func (ort *ORT_SDK) CreateAllocator(session *OrtSession, memInfo *OrtMemoryInfo) (*OrtAllocator, error) {
	var ortAllocator *OrtAllocator
	status := C.CreateAllocator(ort._Api, session, memInfo, &ortAllocator)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}
	return ortAllocator, nil
}

func (ort *ORT_SDK) GetAllocatorWithDefaultOptions() (*OrtAllocator, error) {
	var ortAllocator *OrtAllocator
	status := C.GetAllocatorWithDefaultOptions(ort._Api, &ortAllocator)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}
	return ortAllocator, nil
}

func (ort *ORT_SDK) CreateMemoryInfo(name string, allocatorType OrtAllocatorType, id int,
	mem_type OrtMemType,
) (*OrtMemoryInfo, error) {
	var memInfo *OrtMemoryInfo
	status := C.CreateMemoryInfo(
		ort._Api, C.CString(name),
		C.OrtAllocatorType(allocatorType), C.int(id),
		C.OrtMemType(mem_type),
		&memInfo,
	)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}
	return memInfo, nil
}

func (ort *ORT_SDK) CreateSessionOptions() (*OrtSessionOptions, error) {
	var options *OrtSessionOptions
	status := C.CreateSessionOptions(ort._Api, &options)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}
	return options, nil
}

// func (ort *ORT_SDK) OrtSessionOptionsAppendExecutionProvider_CPU(options *OrtSessionOptions, arena int) error {
// 	status := C.OrtSessionOptionsAppendExecutionProvider_CPU(options, C.int(arena))
// 	return ort.CheckAndReleaseStatus(status)
// }

func (ort *ORT_SDK) CreateCUDAProviderOptions() (*OrtCUDAProviderOptionsV2, error) {
	var options *OrtCUDAProviderOptionsV2
	status := C.CreateCUDAProviderOptions(ort._Api, &options)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}
	return options, nil
}

func (ort *ORT_SDK) UpdateCUDAProviderOptions(cuda_options *OrtCUDAProviderOptionsV2, options ExecutionProviderOptions_CUDA) error {
	status := C.UpdateCUDAProviderOptions(ort._Api, cuda_options, nil, nil, 0)
	if status != nil {
		return ort.CheckAndReleaseStatus(status)
	}
	return nil
}

func (ort *ORT_SDK) SessionOptionsAppendExecutionProvider_CUDA_V2(options *OrtSessionOptions, cuda_options *OrtCUDAProviderOptionsV2) error {
	status := C.SessionOptionsAppendExecutionProvider_CUDA_V2(ort._Api, options, cuda_options)
	if status != nil {
		return ort.CheckAndReleaseStatus(status)
	}
	return nil
}

func (ort *ORT_SDK) CreateCpuMemoryInfo() (*OrtMemoryInfo, error) {
	var memInfo *OrtMemoryInfo
	status := C.CreateCpuMemoryInfo(
		ort._Api,
		C.OrtAllocatorType(OrtDeviceAllocator),
		C.OrtMemType(OrtMemTypeDefault),
		&memInfo,
	)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}
	return memInfo, nil
}

func (ort *ORT_SDK) CreateSessionFromArray(onnxBytes []byte, opt *OrtSessionOptions) (*OrtSession, error) {
	var session *OrtSession
	status := C.CreateSessionFromArray(
		ort._Api, ort._Env,
		unsafe.Pointer(&(onnxBytes[0])), C.size_t(len(onnxBytes)),
		opt, &session,
	)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}

	return session, nil
}

func (ort *ORT_SDK) SessionGetInputCount(session *OrtSession) (int, error) {
	out := C.size_t(0)
	status := C.SessionGetInputCount(ort._Api, session, &out)
	if status != nil {
		return -1, ort.CheckAndReleaseStatus(status)
	}
	return int(uintptr(out)), nil
}

func (ort *ORT_SDK) SessionGetInputName(session *OrtSession, idx int, allocator *OrtAllocator) (string, error) {
	var name *C.char
	status := C.SessionGetInputName(ort._Api, session, C.size_t(idx), allocator, &name)
	if status != nil {
		return "", ort.CheckAndReleaseStatus(status)
	}
	return C.GoString(name), nil
}

func (ort *ORT_SDK) SessionGetInputTypeInfo(session *OrtSession, idx int) (*OrtTypeInfo, error) {
	var info *OrtTypeInfo
	status := C.SessionGetInputTypeInfo(ort._Api, session, C.size_t(idx), &info)
	if status != nil {
		return info, ort.CheckAndReleaseStatus(status)
	}
	return info, nil
}

func (ort *ORT_SDK) SessionGetOutputCount(session *OrtSession) (int, error) {
	out := C.size_t(0)
	status := C.SessionGetOutputCount(ort._Api, session, &out)
	if status != nil {
		return -1, ort.CheckAndReleaseStatus(status)
	}
	return int(uintptr(out)), nil
}

func (ort *ORT_SDK) SessionGetOutputName(session *OrtSession, idx int, allocator *OrtAllocator) (string, error) {
	var name *C.char
	status := C.SessionGetOutputName(ort._Api, session, C.size_t(idx), allocator, &name)
	if status != nil {
		return "", ort.CheckAndReleaseStatus(status)
	}
	return C.GoString(name), nil
}

func (ort *ORT_SDK) SessionGetOutputTypeInfo(session *OrtSession, idx int) (*OrtTypeInfo, error) {
	var info *OrtTypeInfo
	status := C.SessionGetOutputTypeInfo(ort._Api, session, C.size_t(idx), &info)
	if status != nil {
		return info, ort.CheckAndReleaseStatus(status)
	}
	return info, nil
}

func (ort *ORT_SDK) CastTypeInfoToTensorInfo(type_info *OrtTypeInfo) (*OrtTensorTypeAndShapeInfo, error) {
	var out *OrtTensorTypeAndShapeInfo
	status := C.CastTypeInfoToTensorInfo(ort._Api, type_info, &out)
	if status != nil {
		return out, ort.CheckAndReleaseStatus(status)
	}
	return out, nil
}

func (ort *ORT_SDK) GetTensorElementType(info *OrtTensorTypeAndShapeInfo) (ONNXTensorElementDataType, error) {
	var out C.ONNXTensorElementDataType
	status := C.GetTensorElementType(ort._Api, info, &out)
	if status != nil {
		return ONNXTensorElementDataType(out), ort.CheckAndReleaseStatus(status)
	}
	return ONNXTensorElementDataType(out), nil
}

func (ort *ORT_SDK) GetDimensionsCount(info *OrtTensorTypeAndShapeInfo) (int, error) {
	out := C.size_t(0)
	status := C.GetDimensionsCount(ort._Api, info, &out)
	if status != nil {
		return -1, ort.CheckAndReleaseStatus(status)
	}
	return int(uintptr(out)), nil
}

func (ort *ORT_SDK) GetDimensions(info *OrtTensorTypeAndShapeInfo, dim_values_length int) ([]int64, error) {
	// int64_t *p = (int64_t *)malloc(dim_count * sizeof(int64_t));
	var p *C.int64_t = (*C.int64_t)(C.malloc(C.size_t(dim_values_length) * C.size_t(unsafe.Sizeof(int64(0)))))
	defer C.free(unsafe.Pointer(p)) // Defer the free to release the allocated memory

	status := C.GetDimensions(
		ort._Api,
		info,
		p,
		C.size_t(dim_values_length),
	)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}

	data := []int64{}
	for i := 0; i < dim_values_length; i++ {
		// num = *(p + j);
		num := *(*C.int64_t)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*unsafe.Sizeof(*p)))
		data = append(data, int64(num))
	}

	return data, nil
}

func (ort *ORT_SDK) Run(
	session *OrtSession, run_options *OrtRunOptions,
	input_names []string, inputs []*OrtValue,
	output_names []string, outputs []*OrtValue,
) ([]*OrtValue, error) {
	cInputNames := make([]*C.char, len(input_names))
	cOutputNames := make([]*C.char, len(output_names))
	for i, v := range input_names {
		cInputNames[i] = C.CString(v)
	}
	for i, v := range output_names {
		cOutputNames[i] = C.CString(v)
	}

	status := C.Run(
		ort._Api, session, run_options,
		&cInputNames[0], &inputs[0], C.size_t(len(inputs)),
		&cOutputNames[0], C.size_t(len(outputs)), &outputs[0],
	)
	if status != nil {
		return nil, ort.CheckAndReleaseStatus(status)
	}

	return outputs, nil
}

func (ort *ORT_SDK) ReleaseEnv(object *OrtEnv) {
	C.ReleaseEnv(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseMemoryInfo(object *OrtMemoryInfo) {
	C.ReleaseMemoryInfo(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseSession(object *OrtSession) {
	C.ReleaseSession(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseValue(object *OrtValue) {
	C.ReleaseValue(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseRunOptions(object *OrtRunOptions) {
	C.ReleaseRunOptions(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseTypeInfo(object *OrtTypeInfo) {
	C.ReleaseTypeInfo(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseTensorTypeAndShapeInfo(object *OrtTensorTypeAndShapeInfo) {
	C.ReleaseTensorTypeAndShapeInfo(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseSessionOptions(object *OrtSessionOptions) {
	C.ReleaseSessionOptions(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseCustomOpDomain(object *OrtCustomOpDomain) {
	C.ReleaseCustomOpDomain(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseMapTypeInfo(object *OrtMapTypeInfo) {
	C.ReleaseMapTypeInfo(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseSequenceTypeInfo(object *OrtSequenceTypeInfo) {
	C.ReleaseSequenceTypeInfo(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseModelMetadata(object *OrtModelMetadata) {
	C.ReleaseModelMetadata(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseThreadingOptions(object *OrtThreadingOptions) {
	C.ReleaseThreadingOptions(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseAllocator(object *OrtAllocator) {
	C.ReleaseAllocator(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseIoBinding(object *OrtIoBinding) {
	C.ReleaseIoBinding(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseArenaCfg(object *OrtArenaCfg) {
	C.ReleaseArenaCfg(ort._Api, object)
}

func (ort *ORT_SDK) ReleasePrepackedWeightsContainer(object *OrtPrepackedWeightsContainer) {
	C.ReleasePrepackedWeightsContainer(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseOpAttr(object *OrtOpAttr) {
	C.ReleaseOpAttr(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseOp(object *OrtOp) {
	C.ReleaseOp(ort._Api, object)
}

func (ort *ORT_SDK) ReleaseKernelInfo(object *OrtKernelInfo) {
	C.ReleaseKernelInfo(ort._Api, object)
}

func (ort *ORT_SDK) CheckAndReleaseStatus(status *OrtStatus) *OrtStatusError {
	if status == nil {
		return nil
	}
	var msg *C.char
	var code C.OrtErrorCode
	C.CheckAndReleaseStatus(ort._Api, status, &msg, &code)
	return &OrtStatusError{
		Msg:  C.GoString(msg),
		Code: OrtErrorCode(code),
	}
}

// An interface for managing tensors where we don't care about accessing the
// underlying data slice. All typed tensors will support this interface,
// regardless of the underlying data type.
type AnyTensor interface {
	DataType() ONNXTensorElementDataType
	GetShape() Shape
	Destroy() error
	GetInternals() *TensorInternalData
}

var _ AnyTensor = &Tensor[float32]{}

// This wraps internal implementation details to avoid exposing them to users
// via the ArbitraryTensor interface.
type TensorInternalData struct {
	ortValue *OrtValue
}

type Tensor[T TensorData] struct {
	// The shape of the tensor
	shape Shape
	// The go slice containing the flattened data that backs the ONNX tensor.
	data []T
	// The underlying ONNX value we use with the C API.
	ortValue *OrtValue
	session  *Session
}

// Cleans up and frees the memory associated with this tensor.
func (t *Tensor[_]) Destroy() error {
	t.session.sdk.ReleaseValue(t.ortValue)
	t.ortValue = nil
	t.data = nil
	t.shape = nil
	return nil
}

// Returns the slice containing the tensor's underlying data. The contents of
// the slice can be read or written to get or set the tensor's contents.
func (t *Tensor[T]) GetData() []T { return t.data }

func (t *Tensor[T]) GetTensorMutableData() (unsafe.Pointer, error) {
	ptr := unsafe.Pointer(C.NULL) // void*
	status := C.GetTensorMutableData(t.session.sdk._Api, t.ortValue, &ptr)
	if status != nil {
		return ptr, t.session.sdk.CheckAndReleaseStatus(status)
	}
	return ptr, nil
}

// Returns the value from the ONNXTensorElementDataType C enum corresponding to
// the type of data held by this tensor.
func (t *Tensor[T]) DataType() ONNXTensorElementDataType { return GetOnnxTensorElementDataType[T]() }

// Returns the shape of the tensor. The returned shape is only a copy;
// modifying this does *not* change the shape of the underlying tensor.
// (Modifying the tensor's shape can only be accomplished by Destroying and
// recreating the tensor with the same data.)
func (t *Tensor[_]) GetShape() Shape { return t.shape.Clone() }

func (t *Tensor[_]) GetInternals() *TensorInternalData {
	return &TensorInternalData{ortValue: t.ortValue}
}

// Makes a deep copy of the tensor, including its ONNXRuntime value. The Tensor
// returned by this function must be destroyed when no longer needed. The
// returned tensor will also no longer refer to the same underlying data; use
// GetData() to obtain the new underlying slice.
func (t *Tensor[T]) Clone() (*Tensor[T], error) {
	toReturn, e := NewEmptyTensor[T](t.session, t.shape)
	if e != nil {
		return nil, fmt.Errorf("Error allocating tensor clone: %w", e)
	}
	copy(toReturn.GetData(), t.data)
	return toReturn, nil
}

// Creates a new empty tensor with the given shape. The shape provided to this
// function is copied, and is no longer needed after this function returns.
func NewEmptyTensor[T TensorData](session *Session, s Shape) (*Tensor[T], error) {
	e := s.Validate()
	if e != nil {
		return nil, fmt.Errorf("Invalid tensor shape: %w", e)
	}
	elementCount := s.FlattenedSize()
	data := make([]T, elementCount)
	return NewTensor(session, s, data)
}

// Creates a new tensor backed by an existing data slice. The shape provided to
// this function is copied, and is no longer needed after this function
// returns. If the data slice is longer than s.FlattenedSize(), then only the
// first portion of the data will be used.
func NewTensor[T TensorData](session *Session, s Shape, data []T) (*Tensor[T], error) {
	e := s.Validate()
	if e != nil {
		return nil, fmt.Errorf("Invalid tensor shape: %w", e)
	}
	elementCount := s.FlattenedSize()
	if elementCount > int64(len(data)) {
		return nil, fmt.Errorf("The tensor's shape (%s) requires %d "+
			"elements, but only %d were provided\n", s, elementCount,
			len(data))
	}
	var ortValue *OrtValue
	dataType := GetOnnxTensorElementDataType[T]()
	dataSize := unsafe.Sizeof(data[0]) * uintptr(elementCount)

	status := C.CreateTensorWithDataAsOrtValue(
		session.sdk._Api, session.sdk._MemoryInfoPrt,
		unsafe.Pointer(&data[0]),
		C.size_t(dataSize), (*C.int64_t)(unsafe.Pointer(&s[0])),
		C.size_t(len(s)), C.ONNXTensorElementDataType(dataType), &ortValue,
	)
	if status != nil {
		return nil, session.sdk.CheckAndReleaseStatus(status)
	}

	toReturn := Tensor[T]{
		data:     data[0:elementCount],
		shape:    s.Clone(),
		ortValue: ortValue,
		session:  session,
	}
	// TODO: Set a finalizer on new Tensors to hopefully prevent careless
	// memory leaks.
	// - Idea: use a "destroyable" interface?
	return &toReturn, nil
}

func NewInputTensor[T TensorData](session *Session, name string, data []T) (*Tensor[T], error) {
	var s Shape
	for _, v := range session.inputs {
		if v.Name == name || name == "" {
			s = v.Shape
			break
		}
	}
	return NewTensor(session, s, data)
}

func NewEmptyOutputTensor[T TensorData](session *Session, name string) (*Tensor[T], error) {
	var s Shape
	for _, v := range session.outputs {
		if v.Name == name || name == "" {
			s = v.Shape
			break
		}
	}
	return NewEmptyTensor[T](session, s)
}
