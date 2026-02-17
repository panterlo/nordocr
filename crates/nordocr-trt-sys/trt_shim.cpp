// Minimal C++ shim wrapping TensorRT 10.x C++ API as extern "C" functions.
// Compiled by cc crate, dynamically links to nvinfer_10.dll / libnvinfer.so.10.

#include "NvInferRuntime.h"
#include <cstdio>
#include <cstring>

using namespace nvinfer1;

// --- Simple logger that forwards TRT messages to stderr ---

class ShimLogger : public ILogger {
public:
    Severity mMinSeverity = Severity::kWARNING;

    void log(Severity severity, AsciiChar const* msg) noexcept override {
        if (severity > mMinSeverity) return;
        const char* level = "???";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: level = "INTERNAL"; break;
            case Severity::kERROR:          level = "ERROR"; break;
            case Severity::kWARNING:        level = "WARNING"; break;
            case Severity::kINFO:           level = "INFO"; break;
            case Severity::kVERBOSE:        level = "VERBOSE"; break;
        }
        fprintf(stderr, "[TensorRT %s] %s\n", level, msg);
    }
};

static ShimLogger g_logger;

// --- Extern C wrappers ---

extern "C" {

// Create a TensorRT runtime. Returns opaque pointer (IRuntime*).
void* trt_create_runtime() {
    return createInferRuntime(g_logger);
}

// Set logger minimum severity (0=INTERNAL_ERROR .. 4=VERBOSE).
void trt_set_log_severity(int32_t severity) {
    if (severity >= 0 && severity <= 4) {
        g_logger.mMinSeverity = static_cast<ILogger::Severity>(severity);
    }
}

// Destroy a runtime.
void trt_destroy_runtime(void* runtime) {
    if (runtime) {
        static_cast<IRuntime*>(runtime)->~IRuntime();
    }
}

// Deserialize an engine from a byte buffer. Returns opaque pointer (ICudaEngine*).
void* trt_deserialize_engine(void* runtime, const void* data, uint64_t size) {
    if (!runtime || !data || size == 0) return nullptr;
    auto* rt = static_cast<IRuntime*>(runtime);
    return rt->deserializeCudaEngine(data, static_cast<size_t>(size));
}

// Destroy an engine.
void trt_destroy_engine(void* engine) {
    if (engine) {
        static_cast<ICudaEngine*>(engine)->~ICudaEngine();
    }
}

// Get number of I/O tensors.
int32_t trt_engine_get_nb_io_tensors(void* engine) {
    if (!engine) return 0;
    return static_cast<ICudaEngine*>(engine)->getNbIOTensors();
}

// Get tensor name by index. Returns pointer to internal string (do not free).
const char* trt_engine_get_tensor_name(void* engine, int32_t index) {
    if (!engine) return nullptr;
    return static_cast<ICudaEngine*>(engine)->getIOTensorName(index);
}

// Get tensor I/O mode (0=NONE, 1=INPUT, 2=OUTPUT).
int32_t trt_engine_get_tensor_io_mode(void* engine, const char* name) {
    if (!engine || !name) return 0;
    return static_cast<int32_t>(static_cast<ICudaEngine*>(engine)->getTensorIOMode(name));
}

// Get tensor shape. Writes dims to out_dims, returns nb_dims.
int32_t trt_engine_get_tensor_shape(void* engine, const char* name, int64_t* out_dims, int32_t max_dims) {
    if (!engine || !name || !out_dims) return 0;
    Dims d = static_cast<ICudaEngine*>(engine)->getTensorShape(name);
    int32_t n = d.nbDims < max_dims ? d.nbDims : max_dims;
    for (int32_t i = 0; i < n; i++) {
        out_dims[i] = d.d[i];
    }
    return d.nbDims;
}

// Get tensor data type (0=FLOAT, 1=HALF, 2=INT8, 3=INT32, ...).
int32_t trt_engine_get_tensor_dtype(void* engine, const char* name) {
    if (!engine || !name) return -1;
    return static_cast<int32_t>(static_cast<ICudaEngine*>(engine)->getTensorDataType(name));
}

// Create an execution context. Returns opaque pointer (IExecutionContext*).
void* trt_create_execution_context(void* engine) {
    if (!engine) return nullptr;
    return static_cast<ICudaEngine*>(engine)->createExecutionContext();
}

// Destroy an execution context.
void trt_destroy_context(void* context) {
    if (context) {
        static_cast<IExecutionContext*>(context)->~IExecutionContext();
    }
}

// Set input tensor shape (for dynamic shapes).
// Returns 1 on success, 0 on failure.
int32_t trt_context_set_input_shape(void* context, const char* name, const int64_t* dims, int32_t nb_dims) {
    if (!context || !name || !dims || nb_dims <= 0) return 0;
    Dims d;
    d.nbDims = nb_dims;
    for (int32_t i = 0; i < nb_dims && i < Dims::MAX_DIMS; i++) {
        d.d[i] = dims[i];
    }
    return static_cast<IExecutionContext*>(context)->setInputShape(name, d) ? 1 : 0;
}

// Bind a GPU buffer address to a named tensor.
// Returns 1 on success, 0 on failure.
int32_t trt_context_set_tensor_address(void* context, const char* name, void* gpu_ptr) {
    if (!context || !name) return 0;
    return static_cast<IExecutionContext*>(context)->setTensorAddress(name, gpu_ptr) ? 1 : 0;
}

// Enqueue inference on the given CUDA stream.
// stream is a cudaStream_t cast to uint64_t.
// Returns 1 on success, 0 on failure.
int32_t trt_context_enqueue_v3(void* context, uint64_t stream) {
    if (!context) return 0;
    return static_cast<IExecutionContext*>(context)->enqueueV3(
        reinterpret_cast<cudaStream_t>(stream)) ? 1 : 0;
}

// Get library version info.
int32_t trt_get_version_major() { return getInferLibMajorVersion(); }
int32_t trt_get_version_minor() { return getInferLibMinorVersion(); }
int32_t trt_get_version_patch() { return getInferLibPatchVersion(); }
int32_t trt_get_version_build() { return getInferLibBuildVersion(); }

} // extern "C"
