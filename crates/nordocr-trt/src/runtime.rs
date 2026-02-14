use std::path::Path;
use std::ptr;

use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Safe wrapper around a TensorRT runtime + deserialized engine.
pub struct TrtRuntime {
    // In production these would be raw pointers to TensorRT C API objects.
    // We use opaque handles here; the actual FFI calls go through nordocr-trt-sys.
    _runtime_handle: u64,
}

/// A deserialized TensorRT engine ready for inference.
pub struct TrtEngine {
    _engine_handle: u64,
    num_io_tensors: usize,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

/// An execution context bound to an engine for running inference.
pub struct TrtExecutionContext {
    _context_handle: u64,
    engine: TrtEngine,
}

/// Describes a single I/O tensor binding.
#[derive(Debug, Clone)]
pub struct TensorBinding {
    pub name: String,
    pub dims: Vec<i64>,
    pub is_input: bool,
    pub dtype: TrtDataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrtDataType {
    Float32,
    Float16,
    Int8,
    Int32,
    Fp8E4M3,
    Fp4,
}

impl TrtDataType {
    pub fn element_size(&self) -> usize {
        match self {
            TrtDataType::Float32 | TrtDataType::Int32 => 4,
            TrtDataType::Float16 => 2,
            TrtDataType::Int8 | TrtDataType::Fp8E4M3 => 1,
            TrtDataType::Fp4 => 1, // packed, but minimum addressable unit
        }
    }
}

impl TrtRuntime {
    /// Create a TensorRT runtime instance.
    pub fn new() -> Result<Self> {
        // In production: calls nordocr_trt_sys::createInferRuntime(logger)
        // Logger is configured to route TRT messages to tracing.
        tracing::debug!("creating TensorRT runtime");
        Ok(Self {
            _runtime_handle: 0,
        })
    }
}

impl TrtEngine {
    /// Deserialize a TensorRT engine from a file.
    ///
    /// The engine file must have been built for the current GPU architecture
    /// (sm_100 for Blackwell). Use `TrtEngineBuilder` to build engines.
    pub fn load(runtime: &TrtRuntime, path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(OcrError::ModelLoad(format!(
                "engine file not found: {}",
                path.display()
            )));
        }

        tracing::info!(path = %path.display(), "loading TensorRT engine");

        let engine_data = std::fs::read(path)?;

        // In production:
        //   let engine = runtime.deserializeCudaEngine(engine_data.as_ptr(), engine_data.len())
        //   Parse I/O tensor names and shapes from the engine.
        let _ = &engine_data;

        Ok(Self {
            _engine_handle: 0,
            num_io_tensors: 2, // placeholder: 1 input + 1 output
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
        })
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Create an execution context for this engine.
    pub fn create_context(self) -> Result<TrtExecutionContext> {
        // In production: engine->createExecutionContext()
        tracing::debug!("creating TensorRT execution context");
        Ok(TrtExecutionContext {
            _context_handle: 0,
            engine: self,
        })
    }
}

impl TrtExecutionContext {
    /// Set the input tensor shape (for dynamic shapes).
    pub fn set_input_shape(&mut self, name: &str, dims: &[i64]) -> Result<()> {
        tracing::trace!(tensor = name, ?dims, "setting input shape");
        // In production: context->setInputShape(name, Dims{...})
        Ok(())
    }

    /// Bind a GPU buffer to a named I/O tensor.
    pub fn set_tensor_address<T: bytemuck::Pod>(
        &mut self,
        name: &str,
        buffer: &GpuBuffer<T>,
    ) -> Result<()> {
        tracing::trace!(
            tensor = name,
            ptr = format_args!("0x{:x}", buffer.device_ptr()),
            "binding tensor address"
        );
        // In production: context->setTensorAddress(name, buffer.ptr())
        Ok(())
    }

    /// Enqueue inference on the given CUDA stream.
    ///
    /// All bound input tensors must have been filled, and output tensors
    /// must point to sufficiently-sized GPU buffers.
    pub fn enqueue_v3(&self, stream: u64) -> Result<()> {
        tracing::trace!(stream, "enqueuing TensorRT inference");
        // In production: context->enqueueV3(cudaStream_t(stream))
        Ok(())
    }

    pub fn engine(&self) -> &TrtEngine {
        &self.engine
    }
}
