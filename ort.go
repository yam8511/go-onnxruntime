package ort

import (
	"fmt"
	"os"
)

type Input struct {
	name       string                    // Name of the input layer
	input_type ONNXTensorElementDataType // Type of the input layer's elements
	dimensions Shape                     // Shape of the input layer
}

func (input Input) String() string {
	return fmt.Sprintf(
		"input: name: %s, data type = %s, dim: %v\n",
		input.name, input.input_type, input.dimensions,
	)
}

type Output struct {
	name        string                    // Name of the input layer
	output_type ONNXTensorElementDataType // Type of the output layer's elements
	dimensions  Shape                     // Shape of the output layer
}

func (output Output) String() string {
	return fmt.Sprintf(
		"output: name: %s, data type = %s, dim: %v\n",
		output.name, output.output_type, output.dimensions,
	)
}

type Session struct {
	sdk           *ORT_SDK
	session_ptr   *OrtSession
	allocator_ptr *OrtAllocator
	mem_ptr       *OrtMemoryInfo
	inputs        []Input  // ONNX's inputs
	outputs       []Output // ONNX's outputs
	inputsTensor  []*OrtValue
	outputsTensor []*OrtValue
}

func NewSessionWithONNX(sdk *ORT_SDK, onnxFile string, useGPU bool) (*Session, error) {
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
	if useGPU {
		cuda_options, err := sdk.CreateCUDAProviderOptions()
		if err != nil {
			return nil, err
		}
		// err = sdk.UpdateCUDAProviderOptions(cuda_options, OrtCUDAProviderOptions{
		// 	DeviceID:    0,
		// 	GpuMemLimit: math.MaxUint,
		// 	// has_user_compute_stream		_Ctype_int
		// 	// user_compute_stream		unsafe.Pointer
		// 	// default_memory_arena_cfg	*_Ctype_struct_OrtArenaCfg
		// 	// tunable_op_enable		_Ctype_int
		// 	// tunable_op_tuning_enable	_Ctype_int
		// })
		// if err != nil {
		// 	return nil, err
		// }
		err = sdk.SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options)
		if err != nil {
			return nil, err
		}
	}

	{ // 建立 Session
		session.session_ptr, err = sdk.CreateSessionFromArray(onnxBytes, options)
		sdk.ReleaseSessionOptions(options)
		if err != nil {
			return nil, err
		}
	}

	{ // 建立 Allocator
		session.allocator_ptr, err = sdk.GetAllocatorWithDefaultOptions()
		if err != nil {
			defer session.Release()
			return nil, err
		}
	}

	{ // 建立 MemoryInfo
		session.mem_ptr, err = sdk.CreateCpuMemoryInfo()
		if err != nil {
			defer session.Release()
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
				name, err := sdk.SessionGetInputName(session.session_ptr, i, session.allocator_ptr)
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

				session.inputs = append(session.inputs, Input{
					name:       name,
					input_type: dataType,
					dimensions: NewShape(n...),
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
				name, err := sdk.SessionGetOutputName(session.session_ptr, i, session.allocator_ptr)
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
					name:        name,
					output_type: dataType,
					dimensions:  NewShape(n...),
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

func (sess *Session) RunDefault(
	inputData []AnyTensor,
	outputData []AnyTensor,
) error {
	inputNames := []string{}
	for i := range inputData {
		if i < len(sess.inputs) {
			inputInfo := sess.inputs[i]
			inputNames = append(inputNames, inputInfo.name)
		} else {
			return ErrBindingLengthNotEqual
		}
	}

	outputNames := []string{}
	for i := range outputData {
		if i < len(sess.outputs) {
			outputInfo := sess.outputs[i]
			outputNames = append(outputNames, outputInfo.name)
		} else {
			return ErrBindingLengthNotEqual
		}
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
		fmt.Println("ReleaseSession")
		sess.sdk.ReleaseSession(sess.session_ptr)
		fmt.Println("ReleaseSession")
	}
	if sess.allocator_ptr != nil {
		fmt.Println("ReleaseAllocator")
		sess.sdk.ReleaseAllocator(sess.allocator_ptr)
		fmt.Println("ReleaseAllocator2")
	}
	if sess.mem_ptr != nil {
		fmt.Println("ReleaseMemoryInfo")
		sess.sdk.ReleaseMemoryInfo(sess.mem_ptr)
		fmt.Println("ReleaseMemoryInfo2")
	}
}
