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
	_DLL           syscall.Handle
	_ApiBase       *OrtApiBase
	_Api           *OrtApi
	_Env           *OrtEnv
	_AllocatorPtr  *OrtAllocator
	_MemoryInfoPrt *OrtMemoryInfo
	_version       uint32
}

func newOrtApi(opts OrtSdkOption) (*ORT_SDK, error) {
	dll, err := syscall.LoadLibrary(opts.WinDLL_Name)
	if err != nil {
		return nil, err
	}

	sdk := &ORT_SDK{_DLL: dll, _version: opts.Version}
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
	sdk._Api = C.GetApi(sdk._ApiBase, C.uint32_t(opts.Version))
	if sdk._Api == nil {
		sdk.Release()
		return nil, fmt.Errorf("%w: no Ort API return", ErrExportOrtSdk)
	}

	return sdk, nil
}

// Release SDK resource
func (ort *ORT_SDK) Release() {
	if ort._MemoryInfoPrt != nil {
		fmt.Println("ReleaseMemoryInfo")
		ort.ReleaseMemoryInfo(ort._MemoryInfoPrt)
		fmt.Println("ReleaseMemoryInfo2")
	}

	if ort._Env != nil {
		fmt.Println("ReleaseEnv")
		ort.ReleaseEnv(ort._Env)
		fmt.Println("ReleaseEnv2")
	}
	syscall.FreeLibrary(ort._DLL)
}
