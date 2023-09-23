# Go Onnxruntime Wrapper

**features**
- onnx input's and output's names and shapes
- onnx inference
- cuda support in windows and linux

---

## Reference

1. [github.com/nbigaouette/onnxruntime-rs](https://github.com/nbigaouette/onnxruntime-rs)
2. [github.com/yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go)

## Requirements

### Windows

1. Download onnxruntime from https://github.com/microsoft/onnxruntime/releases

    e.g. download file: onnxruntime-win-x64-gpu-1.16.0.zip

### Linux

1. Download onnxruntime from https://github.com/microsoft/onnxruntime/releases

    e.g. download file: onnxruntime-linux-x64-gpu-1.16.0.tgz

2. unpack to `/usr/local/onnxruntime`

    ```shell
    $ tar -zxvf onnxruntime-linux-x64-gpu-1.16.0.tgz
    $ mv onnxruntime-linux-x64-gpu-1.16.0 /usr/local/onnxruntime
    ```

3. set environment

    ```shell
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/onnxruntime/lib
    ```


## Usage

```go
package main

import (
	"log"
	"runtime"

	ort "github.com/yam8511/go-onnxruntime"
)

func main() {
	var onnxruntimeDLL string
	if runtime.GOOS == "windows" {
		onnxruntimeDLL = "path/to/your/onnxruntime.dll"
	}
	sdk, err := ort.New_ORT_SDK(onnxruntimeDLL)
	if err != nil {
		log.Panicln("new onnxruntime sdk error", err)
	}
	defer sdk.Release()

	/*
	 * Load Onnx File
	 */
	session, _ := ort.NewSessionWithONNX(sdk, "path/to/your/model.onnx", true)
	defer session.Release()

	/*
	 * Run session method 1. auto binding onnx input's and output's name and shape
	 */
	{
		inputTensor, _ := ort.NewInputTensor(session, "", []float32{0.1, 0.2, 0.3})
		outputTensor, _ := ort.NewEmptyOutputTensor[float32](session, "")
		_ = session.RunDefault([]ort.AnyTensor{inputTensor}, []ort.AnyTensor{outputTensor})
		_ = outputTensor.GetData()
	}

	/*
	 * Run session method 2. specify tensor's name and shape
	 */
	{
		inputNames := []string{"input"}
		inputTensor, _ := ort.NewTensor(session, ort.NewShape(1, 1, 1), []float32{0.1, 0.2, 0.3})
		outputNames := []string{"output"}
		outputTensor, _ := ort.NewTensor(session, ort.NewShape(1), []float32{})
		_ = session.Run(
			inputNames, []ort.AnyTensor{inputTensor},
			outputNames, []ort.AnyTensor{outputTensor},
		)
		_ = outputTensor.GetData()
	}
}
```

## Example

- [Inference YOLOv8 Object Detection](https://github.com/yam8511/go-onnxruntime-example)
