package ort

import (
	"fmt"
	"log"
	"os"
	"sort"
)

type Input struct {
	Name     string                    // Name of the input layer
	DataType ONNXTensorElementDataType // Type of the input layer's elements
	Shape    Shape                     // Shape of the input layer
}

func (input Input) String() string {
	return fmt.Sprintf(
		"input: name: %s, data type = %s, shape: %v\n",
		input.Name, input.DataType, input.Shape,
	)
}

type Output struct {
	Name     string                    // Name of the input layer
	DataType ONNXTensorElementDataType // Type of the output layer's elements
	Shape    Shape                     // Shape of the output layer
}

func (output Output) String() string {
	return fmt.Sprintf(
		"output: name: %s, data type = %s, shape: %v\n",
		output.Name, output.DataType, output.Shape,
	)
}

type Session struct {
	sdk          *ORT_SDK
	session_ptr  *OrtSession
	metadata_ptr *OrtModelMetadata
	// allocator_ptr *OrtAllocator
	// mem_ptr       *OrtMemoryInfo
	inputs        []Input  // ONNX's inputs
	outputs       []Output // ONNX's outputs
	inputsTensor  []*OrtValue
	outputsTensor []*OrtValue
}

func NewSessionWithONNX(sdk *ORT_SDK, onnxFile string, useGPU bool, deviceIDs ...int) (*Session, error) {
	onnxBytes, err := os.ReadFile(onnxFile)
	if err != nil {
		return nil, err
	}
	session := &Session{sdk: sdk}

	// 建立 Session Options (major in Execution Provider)
	options, err := sdk.CreateSessionOptions()
	if err != nil {
		return nil, err
	}
	defer sdk.ReleaseSessionOptions(options)

	providers, err := sdk.GetAvailableProviders()
	if err != nil {
		return nil, err
	}

	if useGPU {
		cudaAvailable := false
		for _, provider := range providers {
			if provider == "CUDAExecutionProvider" {
				cudaAvailable = true
				break
			}
		}

		if cudaAvailable {
			cuda_options, err := sdk.CreateCUDAProviderOptions()
			if err != nil {
				return nil, err
			}

			stats, err := GPU_stats()
			if err != nil {
				fmt.Println("[warning] cannot get GPU stats:", err)
				stats = []GPU_Stats{}
			}

			deviceID := -1
			for _, id := range deviceIDs {
				deviceID = id
			}
			if len(stats) > 0 {
				sort.SliceStable(stats, func(i, j int) bool {
					return stats[i].Free > stats[j].Free
				})
				for _, stat := range stats {
					if deviceID < 0 {
						deviceID = stat.DeviceID
						fmt.Println("use CUDAExecutionProvider | auto use device =", deviceID, "|", stat.Name)
						break
					} else if stat.DeviceID == deviceID {
						fmt.Println("use CUDAExecutionProvider | device =", deviceID, "|", stat.Name)
						break
					}
				}
			}

			if deviceID >= 0 {
				err = sdk.UpdateCUDAProviderOptions_DeviceID(cuda_options, deviceID)
				if err != nil {
					return nil, err
				}
			}

			err = sdk.SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options)
			sdk.ReleaseCUDAProviderOptions(cuda_options)
			if err != nil {
				return nil, err
			}
		} else {
			log.Println("[warning] CUDAExecutionProvider not available. fallback to CPUExecutionProvider")
		}
	} else {
		fmt.Println("use CPUExecutionProvider")
	}

	err = sdk.DisableMemPattern(options)
	if err != nil {
		return nil, err
	}

	{ // 建立 Session
		session.session_ptr, err = sdk.CreateSessionFromArray(onnxBytes, options)
		if err != nil {
			return nil, err
		}
	}

	{ // 撈取 metadata
		session.metadata_ptr, err = sdk.SessionGetModelMetadata(session.session_ptr)
		if err != nil {
			return nil, err
		}
	}

	{ // 撈取 inputs
		count, err := sdk.SessionGetInputCount(session.session_ptr)
		if err != nil {
			return nil, err
		}

		session.inputs = []Input{}
		for i := 0; i < count; i++ {
			err := func() error {
				name, err := sdk.SessionGetInputName(session.session_ptr, i, sdk._AllocatorPtr)
				// name, err := sdk.SessionGetInputName(session.session_ptr, i, session.allocator_ptr)
				if err != nil {
					return err
				}

				info, err := sdk.SessionGetInputTypeInfo(session.session_ptr, i)
				if err != nil {
					return err
				}
				defer sdk.ReleaseTypeInfo(info)

				tensor, err := sdk.CastTypeInfoToTensorInfo(info)
				if err != nil {
					return err
				}
				dims, err := sdk.GetDimensionsCount(tensor)
				if err != nil {
					return err
				}

				dataType, err := sdk.GetTensorElementType(tensor)
				if err != nil {
					return err
				}

				n, err := sdk.GetDimensions(tensor, dims)
				if err != nil {
					return err
				}
				if len(n) == 0 {
					n = []int64{1}
				}

				session.inputs = append(session.inputs, Input{
					Name:     name,
					DataType: dataType,
					Shape:    NewShape(n...),
				})
				return nil
			}()
			if err != nil {
				return nil, err
			}
		}
	}

	{ // 撈取 outputs
		count, err := sdk.SessionGetOutputCount(session.session_ptr)
		if err != nil {
			return nil, err
		}

		session.outputs = []Output{}
		for i := 0; i < count; i++ {
			err := func() error {
				name, err := sdk.SessionGetOutputName(session.session_ptr, i, sdk._AllocatorPtr)
				// name, err := sdk.SessionGetOutputName(session.session_ptr, i, session.allocator_ptr)
				if err != nil {
					return err
				}

				info, err := sdk.SessionGetOutputTypeInfo(session.session_ptr, i)
				if err != nil {
					return err
				}
				defer sdk.ReleaseTypeInfo(info)

				tensor, err := sdk.CastTypeInfoToTensorInfo(info)
				if err != nil {
					return err
				}
				dims, err := sdk.GetDimensionsCount(tensor)
				if err != nil {
					return err
				}

				dataType, err := sdk.GetTensorElementType(tensor)
				if err != nil {
					return err
				}

				n, err := sdk.GetDimensions(tensor, dims)
				if err != nil {
					return err
				}

				session.outputs = append(session.outputs, Output{
					Name:     name,
					DataType: dataType,
					Shape:    NewShape(n...),
				})
				return nil
			}()
			if err != nil {
				return nil, err
			}
		}
	}

	return session, nil
}

func (sess *Session) Metadata(key string) (string, error) {
	return sess.sdk.ModelMetadataLookupCustomMetadataMap(sess.metadata_ptr, key)
}

func (sess *Session) Outputs() []Output {
	return sess.outputs
}

func (sess *Session) Inputs() []Input {
	return sess.inputs
}

func (sess *Session) Output(name string) (Output, bool) {
	for _, v := range sess.outputs {
		if v.Name == name {
			return v, true
		}
	}
	return Output{}, false
}

func (sess *Session) Input(name string) (Input, bool) {
	for _, v := range sess.inputs {
		if v.Name == name {
			return v, true
		}
	}
	return Input{}, false
}

func (sess *Session) RunDefault(
	inputData []AnyTensor,
	outputData []AnyTensor,
) error {
	inputNames := []string{}
	for i := range inputData {
		inputInfo := sess.inputs[i]
		inputNames = append(inputNames, inputInfo.Name)
	}

	outputNames := []string{}
	for i := range outputData {
		outputInfo := sess.outputs[i]
		outputNames = append(outputNames, outputInfo.Name)
	}

	return sess.Run(inputNames, inputData, outputNames, outputData)
}

func (sess *Session) Run(
	inputNames []string, inputData []AnyTensor,
	outputNames []string, outputData []AnyTensor,
) error {
	if len(inputNames) != len(sess.inputs) || len(outputNames) != len(sess.outputs) ||
		len(inputNames) != len(inputData) || len(outputNames) != len(outputData) {
		return ErrBindingLengthNotEqual
	}

	inputOrtTensors := make([]*OrtValue, len(inputData))
	outputOrtTensors := make([]*OrtValue, len(outputData))
	for i, v := range inputData {
		inputOrtTensors[i] = v.GetInternals().ortValue
	}
	for i, v := range outputData {
		outputOrtTensors[i] = v.GetInternals().ortValue
	}
	sess.inputsTensor = inputOrtTensors
	sess.outputsTensor = outputOrtTensors

	outputOrtTensors, err := sess.sdk.Run(
		sess.session_ptr, nil,
		inputNames, inputOrtTensors,
		outputNames, outputOrtTensors,
	)
	if err != nil {
		return err
	}
	sess.outputsTensor = outputOrtTensors
	return nil
}

func (sess *Session) RunBinding(inputData map[string]AnyTensor, outputData map[string]AnyTensor) error {
	return nil
}

func (sess *Session) Release() {
	if sess.session_ptr != nil {
		sess.sdk.ReleaseSession(sess.session_ptr)
	}
}
