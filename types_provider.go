package ort

type ExecutionProvider int

const (
	ExecutionProvider_CPU = ExecutionProvider(iota)
	ExecutionProvider_GPU
)

type ExecutionProviderOptions interface {
	Options()
}

type ExecutionProviderOptions_CPU struct {
	use_arena bool
}

type ArenaExtendStrategy int

const (
	kNextPowerOfTwo  = ArenaExtendStrategy(0)
	kSameAsRequested = ArenaExtendStrategy(1)
)

type CUDAExecutionProviderCuDNNConvAlgoSearch int

const (
	// expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
	EXHAUSTIVE = CUDAExecutionProviderCuDNNConvAlgoSearch(0)
	// lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
	HEURISTIC = CUDAExecutionProviderCuDNNConvAlgoSearch(1)
	// default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	DEFAULT = CUDAExecutionProviderCuDNNConvAlgoSearch(2)
)

type ExecutionProviderOptions_CUDA struct {
	DeviceID uint
	// The size limit of the device memory arena in bytes. This size limit is only for the execution provider’s arena.
	// The total device memory usage may be higher.
	GpuMemLimit uint
	// The strategy for extending the device memory arena. See [`ArenaExtendStrategy`].
	ArenaExtendStrategy ArenaExtendStrategy
	// ORT leverages cuDNN for convolution operations and the first step in this process is to determine an
	// “optimal” convolution algorithm to use while performing the convolution operation for the given input
	// configuration (input shape, filter shape, etc.) in each `Conv` node. This option controlls the type of search
	// done for cuDNN convolution algorithms. See [`CUDAExecutionProviderCuDNNConvAlgoSearch`] for more info.
	CudnnConvAlgoSearch CUDAExecutionProviderCuDNNConvAlgoSearch
	// Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false,
	// there are race conditions and possibly better performance.
	DoCopyInDefaultStream bool
	// ORT leverages cuDNN for convolution operations and the first step in this process is to determine an
	// “optimal” convolution algorithm to use while performing the convolution operation for the given input
	// configuration (input shape, filter shape, etc.) in each `Conv` node. This sub-step involves querying cuDNN for a
	// “workspace” memory size and have this allocated so that cuDNN can use this auxiliary memory while determining
	// the “optimal” convolution algorithm to use.
	//
	// When `cudnn_conv_use_max_workspace` is false, ORT will clamp the workspace size to 32 MB, which may lead to
	// cuDNN selecting a suboptimal convolution algorithm. The recommended (and default) value is `true`.
	CudnnConvUseMaxWorkspace bool
	// ORT leverages cuDNN for convolution operations. While cuDNN only takes 4-D or 5-D tensors as input for
	// convolution operations, dimension padding is needed if the input is a 3-D tensor. Given an input tensor of shape
	// `[N, C, D]`, it can be padded to `[N, C, D, 1]` or `[N, C, 1, D]`. While both of these padding methods produce
	// the same output, the performance may differ because different convolution algorithms are selected,
	// especially on some devices such as A100. By default, the input is padded to `[N, C, D, 1]`. Set this option to
	// true to instead use `[N, C, 1, D]`.
	CudnnConv1dPadToNc1d bool
	// ORT supports the usage of CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
	// sequentially. To enable the usage of CUDA Graphs, set `enable_cuda_graph` to true.
	// Currently, there are some constraints with regards to using the CUDA Graphs feature:
	//
	// - Models with control-flow ops (i.e. If, Loop and Scan ops) are not supported.
	// - Usage of CUDA Graphs is limited to models where-in all the model ops (graph nodes) can be partitioned to the
	//   CUDA EP.
	// - The input/output types of models must be tensors.
	// - Shapes of inputs/outputs cannot change across inference calls. Dynamic shape models are supported, but the
	//   input/output shapes must be the same across each inference call.
	// - By design, CUDA Graphs is designed to read from/write to the same CUDA virtual memory addresses during the
	//   graph replaying step as it does during the graph capturing step. Due to this requirement, usage of this
	//   feature requires using IOBinding so as to bind memory which will be used as input(s)/output(s) for the CUDA
	//   Graph machinery to read from/write to (please see samples below).
	// - While updating the input(s) for subsequent inference calls, the fresh input(s) need to be copied over to the
	//   corresponding CUDA memory location(s) of the bound `OrtValue` input(s). This is due to the fact that the
	//   “graph replay” will require reading inputs from the same CUDA virtual memory addresses.
	// - Multi-threaded usage is currently not supported, i.e. `run()` MAY NOT be invoked on the same `Session` object
	//   from multiple threads while using CUDA Graphs.
	//
	// > **NOTE**: The very first `run()` performs a variety of tasks under the hood like making CUDA memory
	// > allocations, capturing the CUDA graph for the model, and then performing a graph replay to ensure that the
	// > graph runs. Due to this, the latency associated with the first `run()` is bound to be high. Subsequent
	// > `run()`s only perform graph replays of the graph captured and cached in the first `run()`.
	EnableCudaGraph bool
	// Whether to use strict mode in the `SkipLayerNormalization` implementation. The default and recommanded setting
	// is `false`. If enabled, accuracy may improve slightly, but performance may decrease.
	EnableSkipLayerNormStrictMode bool
}
