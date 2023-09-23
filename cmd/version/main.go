package main

import (
	"fmt"

	ort "github.com/yam8511/go-onnxruntime"
)

func main() {
	sdk, err := ort.New_ORT_SDK(func(opt *ort.OrtSdkOption) {
		opt.WinDLL_Name = "onnxruntime.dll"
		opt.Version = ort.ORT_API_VERSION
		opt.LoggingLevel = ort.ORT_LOGGING_LEVEL_INFO
	})
	if err != nil {
		panic(err)
	}
	defer sdk.Release()

	fmt.Printf("sdk.ORT_API_VERSION(): %v\n", sdk.ORT_API_VERSION())
	fmt.Printf("sdk.GetVersionString(): %v\n", sdk.GetVersionString())
}
