//go:build linux
// +build linux

package ort

/*
#include "onnxruntime_wrapper.h"
#include <stdlib.h>
*/
import "C"

type ORT_SDK struct {
	_ApiBase *OrtApiBase
	_Api     *OrtApi
	_Env     *OrtEnv
}

func newOrtApi(...string) (*ORT_SDK, error) {
	sdk := &ORT_SDK{}
	sdk._ApiBase = C.OrtGetApiBase()
	sdk._Api = C.GetApi(sdk._ApiBase)
	return sdk, nil
}

func (ort *ORT_SDK) Release() {
	if ort._Env != nil {
		ort.ReleaseEnv(ort._Env)
	}
}
