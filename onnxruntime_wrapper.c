#include "onnxruntime_wrapper.h"

/* #region OrtApi */

const OrtApi *GetApi(OrtApiBase *api_base) { return api_base->GetApi(ORT_API_VERSION); }
const char *GetVersionString(OrtApiBase *api_base) { return api_base->GetVersionString(); }

/* #endregion */

/* #region OrtEnv Operation */

/** \brief Create an OrtEnv
 *
 * \param[in] log_severity_level The log severity level.
 * \param[in] logid The log identifier.
 * \param[out] out Returned newly created OrtEnv. Must be freed with OrtApi::ReleaseEnv
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *CreateOrtEnvWithOrtLoggingLevel(OrtApi *ort_api, OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out)
{
  return ort_api->CreateEnv(log_severity_level, logid, out);
}
OrtStatus *CreateOrtEnv(OrtApi *ort_api, const char *logid, OrtEnv **out)
{
  return CreateOrtEnvWithOrtLoggingLevel(ort_api, ORT_LOGGING_LEVEL_WARNING, logid, out);
}

/* #endregion */

/* #region  Allocator Operation */

/** \brief Create an allocator for an ::OrtSession following an ::OrtMemoryInfo
 *
 * \param[in] session
 * \param[in] mem_info valid ::OrtMemoryInfo instance
 * \param[out] out Newly created ::OrtAllocator. Must be freed with OrtApi::ReleaseAllocator
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *CreateAllocator(OrtApi *ort_api, OrtSession *session, OrtMemoryInfo *mem_info, OrtAllocator **out)
{
  return ort_api->CreateAllocator(session, mem_info, out);
}

/** \brief Get the default allocator
 *
 * The default allocator is a CPU based, non-arena. Always returns the same pointer to the same default allocator.
 *
 * \param[out] out Returned value should NOT be freed
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *GetAllocatorWithDefaultOptions(OrtApi *ort_api, OrtAllocator **out)
{
  return ort_api->GetAllocatorWithDefaultOptions(out);
}

/* #endregion */

/* #region  MemoryInfo Operation */

/** \brief Create an ::OrtMemoryInfo
 *
 * \param[in] name
 * \param[in] type
 * \param[in] id
 * \param[in] mem_type
 * \param[out] out Newly created ::OrtMemoryInfo. Must be freed with OrtAPi::ReleaseMemoryInfo
 */
OrtStatus *CreateMemoryInfo(
    OrtApi *ort_api,
    char *name,
    OrtAllocatorType allocator_type, int id,
    OrtMemType mem_type,
    OrtMemoryInfo **out)
{
  return ort_api->CreateMemoryInfo(name, allocator_type, id, mem_type, out);
}

/** \brief Create an ::OrtMemoryInfo for CPU memory
 *
 * Special case version of OrtApi::CreateMemoryInfo for CPU based memory. Same as using OrtApi::CreateMemoryInfo with name = "Cpu" and id = 0.
 *
 * \param[in] type
 * \param[in] mem_type
 * \param[out] out
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *CreateCpuMemoryInfo(OrtApi *ort_api, OrtAllocatorType allocator_type, OrtMemType mem_type, OrtMemoryInfo **out)
{
  return ort_api->CreateCpuMemoryInfo(allocator_type, mem_type, out);
}

/* #endregion */

/* #region SessionOptions Operation */

/** \brief Create an ::OrtSessionOptions object
 *
 * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
 * functions to enable them in the session:<br>
 *   OrtSessionOptionsAppendExecutionProvider_CPU<br>
 *   OrtSessionOptionsAppendExecutionProvider_CUDA<br>
 *   OrtSessionOptionsAppendExecutionProvider_(remaining providers...)<br>
 * The order they are called indicates the preference order as well. In other words call this method
 * on your most preferred execution provider first followed by the less preferred ones.
 * If none are called Ort will use its internal CPU execution provider.
 *
 * \param[out] options The newly created OrtSessionOptions. Must be freed with OrtApi::ReleaseSessionOptions
 */
OrtStatus *CreateSessionOptions(OrtApi *ort_api, OrtSessionOptions **options)
{
  return ort_api->CreateSessionOptions(options);
}

/** \brief Create an OrtCUDAProviderOptionsV2
 *
 * \param[out] out Newly created ::OrtCUDAProviderOptionsV2. Must be released with OrtApi::ReleaseCudaProviderOptions
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 *
 * \since Version 1.11.
 */
OrtStatus *CreateCUDAProviderOptions(OrtApi *ort_api, OrtCUDAProviderOptionsV2 **out)
{
  return ort_api->CreateCUDAProviderOptions(out);
}

/** \brief Set options in a CUDA Execution Provider.
 *
 * Please refer to https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
 * to know the available keys and values. Key should be in null terminated string format of the member of ::OrtCUDAProviderOptionsV2
 * and value should be its related range.
 *
 * For example, key="device_id" and value="0"
 *
 * \param[in] cuda_options
 * \param[in] provider_options_keys Array of UTF-8 null-terminated string for provider options keys
 * \param[in] provider_options_values Array of UTF-8 null-terminated string for provider options values
 * \param[in] num_keys Number of elements in the `provider_option_keys` and `provider_options_values` arrays
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 *
 * \since Version 1.11.
 */
OrtStatus *UpdateCUDAProviderOptions(OrtApi *ort_api, OrtCUDAProviderOptionsV2 *cuda_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
{
  return ort_api->UpdateCUDAProviderOptions(
      cuda_options,
      provider_options_keys,
      provider_options_values,
      num_keys);
}

/** \brief Append CUDA execution provider to the session options
 *
 * If CUDA is not available (due to a non CUDA enabled build), this function will return failure.
 *
 * This is slightly different from OrtApi::SessionOptionsAppendExecutionProvider_CUDA, it takes an
 * ::OrtCUDAProviderOptions which is publicly defined. This takes an opaque ::OrtCUDAProviderOptionsV2
 * which must be created with OrtApi::CreateCUDAProviderOptions.
 *
 * For OrtApi::SessionOptionsAppendExecutionProvider_CUDA, the user needs to instantiate ::OrtCUDAProviderOptions
 * as well as allocate/release buffers for some members of ::OrtCUDAProviderOptions.
 * Here, OrtApi::CreateCUDAProviderOptions and Ortapi::ReleaseCUDAProviderOptions will do the memory management for you.
 *
 * \param[in] options
 * \param[in] cuda_options
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 *
 * \since Version 1.11.
 */
OrtStatus *SessionOptionsAppendExecutionProvider_CUDA_V2(OrtApi *ort_api, OrtSessionOptions *options, const OrtCUDAProviderOptionsV2 *cuda_options)
{
  return ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options);
}

/* #endregion */

/* #region Session Operation */

/** \brief Create an OrtSession from memory
 *
 * \param[in] env
 * \param[in] model_data
 * \param[in] model_data_length
 * \param[in] options
 * \param[out] out Returned newly created OrtSession. Must be freed with OrtApi::ReleaseSession
 */
OrtStatus *
CreateSessionFromArray(
    OrtApi *ort_api,
    OrtEnv *env, void *model_data, size_t model_data_length,
    OrtSessionOptions *options,
    OrtSession **out)
{
  return ort_api->CreateSessionFromArray(env, model_data, model_data_length, options, out);
}

/** \brief Create an OrtSession from a model file
 *
 * \param[in] env
 * \param[in] model_path
 * \param[in] options
 * \param[out] out Returned newly created OrtSession. Must be freed with OrtApi::ReleaseSession
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
// TODO: document the path separator convention? '/' vs '\'
// TODO: should specify the access characteristics of model_path. Is this read only during the
// execution of CreateSession, or does the OrtSession retain a handle to the file/directory
// and continue to access throughout the OrtSession lifetime?
//  What sort of access is needed to model_path : read or read/write?
OrtStatus *CreateSession(
    OrtApi *ort_api,
    const OrtEnv *env, const ORTCHAR_T *model_path,
    const OrtSessionOptions *options, OrtSession **out)
{
  return ort_api->CreateSession(env, model_path, options, out);
}

/** \brief Get input count for a session
 *
 * This number must also match the number of inputs passed to OrtApi::Run
 *
 * \see OrtApi::SessionGetInputTypeInfo, OrtApi::SessionGetInputName, OrtApi::Session
 *
 * \param[in] session
 * \param[out] out Number of inputs
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *SessionGetInputCount(OrtApi *ort_api, OrtSession *session, size_t *out)
{
  return ort_api->SessionGetInputCount(session, out);
}

/** \brief Get input name
 *
 * \param[in] session
 * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetInputCount returns (exclusive)
 * \param[in] allocator
 * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *SessionGetInputName(OrtApi *ort_api, OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
{
  return ort_api->SessionGetInputName(session, index, allocator, value);
}

/** \brief Get input type information
 *
 * \param[in] session
 * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetInputCount returns (exclusive)
 * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *SessionGetInputTypeInfo(OrtApi *ort_api, OrtSession *session, size_t index, OrtTypeInfo **type_info)
{
  return ort_api->SessionGetInputTypeInfo(session, index, type_info);
}

/** \brief Get output count for a session
 *
 * This number must also match the number of outputs returned by OrtApi::Run
 *
 * \see OrtApi::SessionGetOutputTypeInfo, OrtApi::SessionGetOutputName, OrtApi::Session
 *
 * \param[in] session
 * \param[out] out Number of outputs
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *SessionGetOutputCount(OrtApi *ort_api, OrtSession *session, size_t *out)
{
  return ort_api->SessionGetOutputCount(session, out);
}

/** \brief Get output name
 *
 * \param[in] session
 * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOutputCount returns (exclusive)
 * \param[in] allocator
 * \param[out] value Set to a null terminated UTF-8 encoded string allocated using `allocator`. Must be freed using `allocator`.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *SessionGetOutputName(OrtApi *ort_api, OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
{
  return ort_api->SessionGetOutputName(session, index, allocator, value);
}

/** \brief Get output type information
 *
 * \param[in] session
 * \param[in] index Must be between 0 (inclusive) and what OrtApi::SessionGetOutputCount returns (exclusive)
 * \param[out] type_info Must be freed with OrtApi::ReleaseTypeInfo
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *SessionGetOutputTypeInfo(OrtApi *ort_api, OrtSession *session, size_t index, OrtTypeInfo **type_info)
{
  return ort_api->SessionGetOutputTypeInfo(session, index, type_info);
}

/** \brief Get ::OrtTensorTypeAndShapeInfo from an ::OrtTypeInfo
 *
 * \param[in] type_info
 * \param[out] out Do not free this value, it will be valid until type_info is freed.
 *             If type_info does not represent tensor, this value will be set to nullptr.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *CastTypeInfoToTensorInfo(OrtApi *ort_api, OrtTypeInfo *type_info, const OrtTensorTypeAndShapeInfo **out)
{
  return ort_api->CastTypeInfoToTensorInfo(type_info, out);
}

/** \brief Get element type in ::OrtTensorTypeAndShapeInfo
 *
 * \see OrtApi::SetTensorElementType
 *
 * \param[in] info
 * \param[out] out
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *GetTensorElementType(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *info, ONNXTensorElementDataType *out)
{
  return ort_api->GetTensorElementType(info, out);
}

/** \brief Get dimension count in ::OrtTensorTypeAndShapeInfo
 *
 * \see OrtApi::GetDimensions
 *
 * \param[in] info
 * \param[out] out
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *GetDimensionsCount(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *info, size_t *out)
{
  return ort_api->GetDimensionsCount(info, out);
}

/** \brief Get dimensions in ::OrtTensorTypeAndShapeInfo
 *
 * \param[in] info
 * \param[out] dim_values Array with `dim_values_length` elements. On return, filled with the dimensions stored in the ::OrtTensorTypeAndShapeInfo
 * \param[in] dim_values_length Number of elements in `dim_values`. Use OrtApi::GetDimensionsCount to get this value
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *GetDimensions(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length)
{
  return ort_api->GetDimensions(info, dim_values, dim_values_length);
}

/** \brief Create a tensor backed by a user supplied buffer
 *
 * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
 * p_data is owned by caller. ReleaseValue won't release p_data.
 *
 * \param[in] info Memory description of where the p_data buffer resides (CPU vs GPU etc).
 * \param[in] p_data Pointer to the data buffer.
 * \param[in] p_data_len The number of bytes in the data buffer.
 * \param[in] shape Pointer to the tensor shape dimensions.
 * \param[in] shape_len The number of tensor shape dimensions.
 * \param[in] type The data type.
 * \param[out] out Returns newly created ::OrtValue. Must be freed with OrtApi::ReleaseValue
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *CreateTensorWithDataAsOrtValue(
    OrtApi *ort_api, const OrtMemoryInfo *info,
    void *p_data, size_t p_data_len,
    const int64_t *shape, size_t shape_len,
    ONNXTensorElementDataType type, OrtValue **out)
{
  return ort_api->CreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape, shape_len, type, out);
}

/** \brief Run the model in an ::OrtSession
 *
 * Will not return until the model run has completed. Multiple threads might be used to run the model based on
 * the options in the ::OrtSession and settings used when creating the ::OrtEnv
 *
 * \param[in] session
 * \param[in] run_options If nullptr, will use a default ::OrtRunOptions
 * \param[in] input_names Array of null terminated UTF8 encoded strings of the input names
 * \param[in] inputs Array of ::OrtValue%s of the input values
 * \param[in] input_len Number of elements in the input_names and inputs arrays
 * \param[in] output_names Array of null terminated UTF8 encoded strings of the output names
 * \param[in] output_names_len Number of elements in the output_names and outputs array
 * \param[out] outputs Array of ::OrtValue%s that the outputs are stored in. This can also be
 *     an array of nullptr values, in this case ::OrtValue objects will be allocated and pointers
 *     to them will be set into the `outputs` array.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *Run(
    OrtApi *ort_api, OrtSession *session, const OrtRunOptions *run_options,
    const char *const *input_names, const OrtValue *const *inputs, size_t input_len,
    const char *const *output_names, size_t output_names_len, OrtValue **outputs)
{
  return ort_api->Run(
      session, run_options,
      input_names, inputs, input_len,
      output_names, output_names_len, outputs);
}

/** \brief Create an ::OrtIoBinding instance
 *
 * An IoBinding object allows one to bind pre-allocated ::OrtValue%s to input names.
 * Thus if you want to use a raw on device buffer as input or output you can avoid
 * extra copy during runtime.
 *
 * \param[in] session
 * \param[out] out Newly created ::OrtIoBinding. Must be freed with OrtApi::ReleaseIoBinding
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *CreateIoBinding(OrtApi *ort_api, OrtSession *session, OrtIoBinding **out)
{
  return ort_api->CreateIoBinding(session, out);
}

/** \brief Bind an ::OrtValue to an ::OrtIoBinding input
 *
 * When using OrtApi::RunWithBinding this value is used for the named input
 *
 * \param[in] binding_ptr
 * \param[in] name Name for the model input
 * \param[in] val_ptr ::OrtValue of Tensor type.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *BindInput(OrtApi *ort_api, OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr)
{
  return ort_api->BindInput(binding_ptr, name, val_ptr);
}

/** \brief Bind an ::OrtValue to an ::OrtIoBinding output
 *
 * When using OrtApi::RunWithBinding this value is used for the named output
 *
 * \param[in] binding_ptr
 * \param[in] name Null terminated string of the model output name
 * \param[in] val_ptr ::OrtValue of Tensor type.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *BindOutput(OrtApi *ort_api, OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr)
{
  return ort_api->BindOutput(binding_ptr, name, val_ptr);
}

/** \brief Bind an ::OrtIoBinding output to a device
 *
 * Binds the ::OrtValue to a device which is specified by ::OrtMemoryInfo.
 * You can either create an instance of ::OrtMemoryInfo with a device id or obtain one from the allocator that you have created/are using
 * This is useful when one or more outputs have dynamic shapes and, it is hard to pre-allocate and bind a chunk of
 * memory within ::OrtValue ahead of time.
 *
 * \see OrtApi::RunWithBinding
 *
 * \param[in] binding_ptr
 * \param[in] name Null terminated string of the device name
 * \param[in] mem_info_ptr
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *BindOutputToDevice(OrtApi *ort_api, OrtIoBinding *binding_ptr, const char *name, const OrtMemoryInfo *mem_info_ptr)
{
  return ort_api->BindOutputToDevice(binding_ptr, name, mem_info_ptr);
}

/** \brief Run a model using Io Bindings for the inputs & outputs
 *
 * \see OrtApi::Run
 *
 * \param[in] session
 * \param[in] run_options
 * \param[in] binding_ptr
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
OrtStatus *RunWithBinding(OrtApi *ort_api, OrtSession *session, const OrtRunOptions *run_options, const OrtIoBinding *binding_ptr)
{
  return ort_api->RunWithBinding(session, run_options, binding_ptr);
}

/* #endregion */

/* #region Release Operation */

void ReleaseEnv(OrtApi *ort_api, OrtEnv *object)
{
  ort_api->ReleaseEnv(object);
}
void ReleaseStatus(OrtApi *ort_api, OrtStatus *object) { ort_api->ReleaseStatus(object); }
void ReleaseMemoryInfo(OrtApi *ort_api, OrtMemoryInfo *object) { ort_api->ReleaseMemoryInfo(object); }
// Don't call ReleaseSession from Dllmain (because session owns a thread pool)
void ReleaseSession(OrtApi *ort_api, OrtSession *object) { ort_api->ReleaseSession(object); }
void ReleaseValue(OrtApi *ort_api, OrtValue *object) { ort_api->ReleaseValue(object); }
void ReleaseRunOptions(OrtApi *ort_api, OrtRunOptions *object) { ort_api->ReleaseRunOptions(object); }
void ReleaseTypeInfo(OrtApi *ort_api, OrtTypeInfo *object) { ort_api->ReleaseTypeInfo(object); }
void ReleaseTensorTypeAndShapeInfo(OrtApi *ort_api, OrtTensorTypeAndShapeInfo *object) { ort_api->ReleaseTensorTypeAndShapeInfo(object); }
void ReleaseSessionOptions(OrtApi *ort_api, OrtSessionOptions *object) { ort_api->ReleaseSessionOptions(object); }
void ReleaseCustomOpDomain(OrtApi *ort_api, OrtCustomOpDomain *object) { ort_api->ReleaseCustomOpDomain(object); }
void ReleaseMapTypeInfo(OrtApi *ort_api, OrtMapTypeInfo *object) { ort_api->ReleaseMapTypeInfo(object); }
void ReleaseSequenceTypeInfo(OrtApi *ort_api, OrtSequenceTypeInfo *object) { ort_api->ReleaseSequenceTypeInfo(object); }
void ReleaseModelMetadata(OrtApi *ort_api, OrtModelMetadata *object) { ort_api->ReleaseModelMetadata(object); }
void ReleaseThreadingOptions(OrtApi *ort_api, OrtThreadingOptions *object) { ort_api->ReleaseThreadingOptions(object); }
void ReleaseAllocator(OrtApi *ort_api, OrtAllocator *object) { ort_api->ReleaseAllocator(object); }
void ReleaseIoBinding(OrtApi *ort_api, OrtIoBinding *object) { ort_api->ReleaseIoBinding(object); }
void ReleaseArenaCfg(OrtApi *ort_api, OrtArenaCfg *object) { ort_api->ReleaseArenaCfg(object); }
void ReleasePrepackedWeightsContainer(OrtApi *ort_api, OrtPrepackedWeightsContainer *object) { ort_api->ReleasePrepackedWeightsContainer(object); }
void ReleaseOpAttr(OrtApi *ort_api, OrtOpAttr *object) { ort_api->ReleaseOpAttr(object); }
void ReleaseOp(OrtApi *ort_api, OrtOp *object) { ort_api->ReleaseOp(object); }
void ReleaseKernelInfo(OrtApi *ort_api, OrtKernelInfo *object) { ort_api->ReleaseKernelInfo(object); }
void CheckAndReleaseStatus(OrtApi *ort_api, OrtStatus *status, const char **msg, OrtErrorCode *code)
{
  *msg = ort_api->GetErrorMessage(status);
  *code = ort_api->GetErrorCode(status);
  ort_api->ReleaseStatus(status);
}

/* #endregion */
