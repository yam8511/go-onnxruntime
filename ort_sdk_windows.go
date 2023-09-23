//go:build windows
// +build windows

package ort

/*
#include "onnxruntime_wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"syscall"
	"unsafe"
)

type ORT_SDK struct {
	_DLL     syscall.Handle
	_ApiBase *OrtApiBase
	_Api     *OrtApi
	_Env     *OrtEnv
}

func newOrtApi(dllNames ...string) (*ORT_SDK, error) {
	dllName := "onnxruntime.dll"
	for _, v := range dllNames {
		dllName = v
	}

	dll, err := syscall.LoadLibrary(dllName)
	if err != nil {
		return nil, err
	}

	sdk := &ORT_SDK{_DLL: dll}
	getApiBaseProc, e := syscall.GetProcAddress(dll, "OrtGetApiBase")
	if e != nil {
		sdk.Release()
		return nil, fmt.Errorf("%w: %w", ErrExportOrtSdk, e)
	}

	ortApiBase, _, e := syscall.SyscallN(uintptr(getApiBaseProc), 0)
	if ortApiBase == 0 {
		sdk.Release()
		if e != nil {
			return nil, fmt.Errorf("%w: %w", ErrExportOrtSdk, e)
		} else {
			return nil, fmt.Errorf("%w: calling no result", ErrExportOrtSdk)
		}
	}
	sdk._ApiBase = (*OrtApiBase)(unsafe.Pointer(ortApiBase))

	// Initialize Ort API
	sdk._Api = C.GetApi(sdk._ApiBase)
	if sdk._Api == nil {
		sdk.Release()
		return nil, fmt.Errorf("%w: no Ort API return", ErrExportOrtSdk)
	}

	return sdk, nil
}

// Release SDK resource
func (ort *ORT_SDK) Release() {
	if ort._Env != nil {
		ort.ReleaseEnv(ort._Env)
	}
	syscall.FreeLibrary(ort._DLL)
}
