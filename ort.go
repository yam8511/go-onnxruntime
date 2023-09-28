package ort

import (
	"fmt"
	"os"
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
	sdk         *ORT_SDK
	session_ptr *OrtSession
	// allocator_ptr *OrtAllocator
	// mem_ptr       *OrtMemoryInfo
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
