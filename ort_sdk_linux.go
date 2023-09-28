//go:build linux
// +build linux

package ort

/*
#include "onnxruntime_wrapper.h"
#include <stdlib.h>
*/
import "C"

type ORT_SDK struct {
	_ApiBase       *OrtApiBase
	_Api           *OrtApi
	_Env           *OrtEnv
	_AllocatorPtr  *OrtAllocator
	_MemoryInfoPrt *OrtMemoryInfo
	_version       uint32
}

func newOrtApi(opts OrtSdkOption) (*ORT_SDK, error) {
	sdk := &ORT_SDK{_version: opts.Version}
	sdk._ApiBase = C.OrtGetApiBase()
	sdk._Api = C.GetApi(sdk._ApiBase, C.uint32_t(opts.Version))
	return sdk, nil
}

func (ort *ORT_SDK) Release() {
	if ort._MemoryInfoPrt != nil {
		ort.ReleaseMemoryInfo(ort._MemoryInfoPrt)
	}

	if ort._Env != nil {
		ort.ReleaseEnv(ort._Env)
	}
}
