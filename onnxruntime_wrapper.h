// #ifndef ONNXRUNTIME_WRAPPER_H
// #define ONNXRUNTIME_WRAPPER_H

// We want to always use the unix-like onnxruntime C APIs, even on Windows, so
// we need to undefine _WIN32 before including onnxruntime_c_api.h. However,
// this requires a careful song-and-dance.

// First, include these common headers, as they get transitively included by
// onnxruntime_c_api.h. We need to include them ourselves, first, so that the
// preprocessor will skip then while _WIN32 is undefined.
#include <stdio.h>
#include <stdlib.h>

// Next, we actually include the header.
#undef _WIN32
#include "onnxruntime_c_api.h"

// ... However, mingw will complain if _WIN32 is *not* defined! So redefine it.
#define _WIN32

#ifdef __cplusplus
extern "C"
{
#endif

    /* #region OrtApi */
    const char *GetVersionString(OrtApiBase *api_base);
    const OrtApi *GetApi(OrtApiBase *api_base);
    /* #endregion */

    /* #region OrtEnv Operation */
    OrtStatus *CreateOrtEnvWithOrtLoggingLevel(OrtApi *ort_api, OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out);
    OrtStatus *CreateOrtEnv(OrtApi *ort_api, const char *logid, OrtEnv **out);
    /* #endregion */

    /* #region Allocator Operation */
    OrtStatus *CreateAllocator(OrtApi *ort_api, OrtSession *session, OrtMemoryInfo *mem_info, OrtAllocator **out);
    OrtStatus *GetAllocatorWithDefaultOptions(OrtApi *ort_api, OrtAllocator **out);
    /* #endregion */

    /* #region MemoryInfo Operation */
    OrtStatus *CreateMemoryInfo(
        OrtApi *ort_api,
        char *name,
        OrtAllocatorType allocator_type, int id,
        OrtMemType mem_type,
        OrtMemoryInfo **out);
    OrtStatus *CreateCpuMemoryInfo(OrtApi *ort_api, OrtAllocatorType allocator_type, OrtMemType mem_type, OrtMemoryInfo **out);
    /* #endregion */

    /* #region SessionOptions Operation */
    OrtStatus *CreateSessionOptions(OrtApi *ort_api, OrtSessionOptions **options);
    OrtStatus *CreateCUDAProviderOptions(OrtApi *ort_api, OrtCUDAProviderOptionsV2 **out);
    OrtStatus *UpdateCUDAProviderOptions(OrtApi *ort_api, OrtCUDAProviderOptionsV2 *cuda_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys);
    OrtStatus *SessionOptionsAppendExecutionProvider_CUDA_V2(OrtApi *ort_api, OrtSessionOptions *options, const OrtCUDAProviderOptionsV2 *cuda_options);
    /* #endregion */

    /* #region Session Operation */
    OrtStatus *CreateSessionFromArray(OrtApi *ort_api, OrtEnv *env, void *model_data, size_t model_data_length, OrtSessionOptions *options, OrtSession **out);
    OrtStatus *CreateSession(
        OrtApi *ort_api,
        const OrtEnv *env, const ORTCHAR_T *model_path,
        const OrtSessionOptions *options, OrtSession **out);
    OrtStatus *SessionGetInputCount(OrtApi *ort_api, OrtSession *session, size_t *out);
    OrtStatus *SessionGetInputName(OrtApi *ort_api, OrtSession *session, size_t index, OrtAllocator *allocator, char **value);
    OrtStatus *SessionGetInputTypeInfo(OrtApi *ort_api, OrtSession *session, size_t index, OrtTypeInfo **type_info);
    OrtStatus *SessionGetOutputCount(OrtApi *ort_api, OrtSession *session, size_t *out);
    OrtStatus *SessionGetOutputName(OrtApi *ort_api, OrtSession *session, size_t index, OrtAllocator *allocator, char **value);
    OrtStatus *SessionGetOutputTypeInfo(OrtApi *ort_api, OrtSession *session, size_t index, OrtTypeInfo **type_info);
    OrtStatus *CastTypeInfoToTensorInfo(OrtApi *ort_api, OrtTypeInfo *type_info, const OrtTensorTypeAndShapeInfo **out);
    OrtStatus *GetTensorElementType(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *info, ONNXTensorElementDataType *out);
    OrtStatus *GetDimensionsCount(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *info, size_t *out);
    OrtStatus *GetDimensions(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length);
    OrtStatus *CreateTensorWithDataAsOrtValue(
        OrtApi *ort_api, const OrtMemoryInfo *info,
        void *p_data, size_t p_data_len,
        const int64_t *shape, size_t shape_len,
        ONNXTensorElementDataType type, OrtValue **out);
    OrtStatus *Run(
        OrtApi *ort_api, OrtSession *session, const OrtRunOptions *run_options,
        const char *const *input_names, const OrtValue *const *inputs, size_t input_len,
        const char *const *output_names, size_t output_names_len, OrtValue **outputs);
    OrtStatus *CreateIoBinding(OrtApi *ort_api, OrtSession *session, OrtIoBinding **out);
    OrtStatus *BindInput(OrtApi *ort_api, OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr);
    OrtStatus *BindOutput(OrtApi *ort_api, OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr);
    OrtStatus *BindOutputToDevice(OrtApi *ort_api, OrtIoBinding *binding_ptr, const char *name, const OrtMemoryInfo *mem_info_ptr);
    OrtStatus *RunWithBinding(OrtApi *ort_api, OrtSession *session, const OrtRunOptions *run_options, const OrtIoBinding *binding_ptr);
    /* #endregion */

    /* #region Release Operation */
    void ReleaseEnv(OrtApi *ort_api, OrtEnv *object);
    void ReleaseStatus(OrtApi *ort_api, OrtStatus *object);
    void ReleaseMemoryInfo(OrtApi *ort_api, OrtMemoryInfo *object);
    void ReleaseSession(OrtApi *ort_api, OrtSession *object);
    void ReleaseValue(OrtApi *ort_api, OrtValue *object);
    void ReleaseRunOptions(OrtApi *ort_api, OrtRunOptions *object);
    void ReleaseTypeInfo(OrtApi *ort_api, OrtTypeInfo *object);
    void ReleaseTensorTypeAndShapeInfo(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *object);
    void ReleaseSessionOptions(OrtApi *ort_api, OrtSessionOptions *object);
    void ReleaseCustomOpDomain(OrtApi *ort_api, OrtCustomOpDomain *object);
    void ReleaseMapTypeInfo(OrtApi *ort_api, OrtMapTypeInfo *object);
    void ReleaseSequenceTypeInfo(OrtApi *ort_api, OrtSequenceTypeInfo *object);
    void ReleaseModelMetadata(OrtApi *ort_api, OrtModelMetadata *object);
    void ReleaseThreadingOptions(OrtApi *ort_api, OrtThreadingOptions *object);
    void ReleaseAllocator(OrtApi *ort_api, OrtAllocator *object);
    void ReleaseIoBinding(OrtApi *ort_api, OrtIoBinding *object);
    void ReleaseArenaCfg(OrtApi *ort_api, OrtArenaCfg *object);
    void ReleasePrepackedWeightsContainer(OrtApi *ort_api, OrtPrepackedWeightsContainer *object);
    void ReleaseOpAttr(OrtApi *ort_api, OrtOpAttr *object);
    void ReleaseOp(OrtApi *ort_api, OrtOp *object);
    void ReleaseKernelInfo(OrtApi *ort_api, OrtKernelInfo *object);
    void CheckAndReleaseStatus(OrtApi *ort_api, OrtStatus *status, const char **msg, OrtErrorCode *code);
    /* #endregion */

#ifdef __cplusplus
} // extern "C"
#endif
// #endif // ONNXRUNTIME_WRAPPER_H
