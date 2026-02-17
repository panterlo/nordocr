use std::ffi::{c_void, CStr, CString};
use std::path::Path;

use nordocr_core::{OcrError, Result};
use nordocr_gpu::GpuBuffer;
use nordocr_trt_sys::{self, TrtDtype, TrtTensorIOMode};

/// Safe wrapper around a TensorRT runtime.
pub struct TrtRuntime {
    handle: *mut c_void,
}

// TRT runtime/engine/context handles are thread-safe for non-overlapping operations.
unsafe impl Send for TrtRuntime {}

impl TrtRuntime {
    /// Create a TensorRT runtime instance.
    pub fn new() -> Result<Self> {
        let handle = unsafe { nordocr_trt_sys::trt_create_runtime() };
        if handle.is_null() {
            return Err(OcrError::TensorRt("failed to create TRT runtime".into()));
        }

        // Log the TRT version.
        let major = unsafe { nordocr_trt_sys::trt_get_version_major() };
        let minor = unsafe { nordocr_trt_sys::trt_get_version_minor() };
        let patch = unsafe { nordocr_trt_sys::trt_get_version_patch() };
        tracing::info!(version = %format!("{major}.{minor}.{patch}"), "TensorRT runtime created");

        Ok(Self { handle })
    }

    /// Set log severity (0=INTERNAL_ERROR, 1=ERROR, 2=WARNING, 3=INFO, 4=VERBOSE).
    pub fn set_log_severity(&self, severity: i32) {
        unsafe { nordocr_trt_sys::trt_set_log_severity(severity) };
    }
}

impl Drop for TrtRuntime {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nordocr_trt_sys::trt_destroy_runtime(self.handle) };
        }
    }
}

/// A deserialized TensorRT engine ready for inference.
pub struct TrtEngine {
    handle: *mut c_void,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_dtypes: Vec<TrtDataType>,
    output_dtypes: Vec<TrtDataType>,
}

unsafe impl Send for TrtEngine {}

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
            TrtDataType::Fp4 => 1,
        }
    }

    fn from_trt(dtype: TrtDtype) -> Self {
        match dtype {
            TrtDtype::Float => Self::Float32,
            TrtDtype::Half => Self::Float16,
            TrtDtype::Int8 => Self::Int8,
            TrtDtype::Int32 => Self::Int32,
            TrtDtype::Fp8E4M3 => Self::Fp8E4M3,
            TrtDtype::Fp4 => Self::Fp4,
            _ => Self::Float32, // fallback
        }
    }
}

impl TrtEngine {
    /// Deserialize a TensorRT engine from a file.
    pub fn load(runtime: &TrtRuntime, path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(OcrError::ModelLoad(format!(
                "engine file not found: {}",
                path.display()
            )));
        }

        tracing::info!(path = %path.display(), "loading TensorRT engine");

        let engine_data = std::fs::read(path)?;

        let handle = unsafe {
            nordocr_trt_sys::trt_deserialize_engine(
                runtime.handle,
                engine_data.as_ptr() as *const c_void,
                engine_data.len() as u64,
            )
        };

        if handle.is_null() {
            return Err(OcrError::ModelLoad(format!(
                "failed to deserialize engine: {}",
                path.display()
            )));
        }

        // Enumerate I/O tensors.
        let num_io = unsafe { nordocr_trt_sys::trt_engine_get_nb_io_tensors(handle) };
        let mut input_names = Vec::new();
        let mut output_names = Vec::new();
        let mut input_dtypes = Vec::new();
        let mut output_dtypes = Vec::new();

        for i in 0..num_io {
            let name_ptr = unsafe { nordocr_trt_sys::trt_engine_get_tensor_name(handle, i) };
            if name_ptr.is_null() {
                continue;
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            let name_c = CString::new(name.as_str()).unwrap();
            let io_mode_i = unsafe {
                nordocr_trt_sys::trt_engine_get_tensor_io_mode(handle, name_c.as_ptr())
            };
            let dtype_i = unsafe {
                nordocr_trt_sys::trt_engine_get_tensor_dtype(handle, name_c.as_ptr())
            };
            let dtype = TrtDtype::from_i32(dtype_i)
                .map(TrtDataType::from_trt)
                .unwrap_or(TrtDataType::Float32);

            // Get shape for logging.
            let mut dims = [0i64; 8];
            let nb_dims = unsafe {
                nordocr_trt_sys::trt_engine_get_tensor_shape(
                    handle,
                    name_c.as_ptr(),
                    dims.as_mut_ptr(),
                    8,
                )
            };
            let shape: Vec<i64> = dims[..nb_dims as usize].to_vec();

            match TrtTensorIOMode::from_i32(io_mode_i) {
                Some(TrtTensorIOMode::Input) => {
                    tracing::debug!(tensor = %name, ?shape, ?dtype, "input tensor");
                    input_names.push(name);
                    input_dtypes.push(dtype);
                }
                Some(TrtTensorIOMode::Output) => {
                    tracing::debug!(tensor = %name, ?shape, ?dtype, "output tensor");
                    output_names.push(name);
                    output_dtypes.push(dtype);
                }
                _ => {}
            }
        }

        tracing::info!(
            inputs = ?input_names,
            outputs = ?output_names,
            "engine loaded"
        );

        Ok(Self {
            handle,
            input_names,
            output_names,
            input_dtypes,
            output_dtypes,
        })
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    pub fn input_dtypes(&self) -> &[TrtDataType] {
        &self.input_dtypes
    }

    pub fn output_dtypes(&self) -> &[TrtDataType] {
        &self.output_dtypes
    }

    /// Create an execution context for this engine.
    pub fn create_context(self) -> Result<TrtExecutionContext> {
        let ctx_handle = unsafe {
            nordocr_trt_sys::trt_create_execution_context(self.handle)
        };
        if ctx_handle.is_null() {
            return Err(OcrError::TensorRt("failed to create execution context".into()));
        }
        tracing::debug!("created TensorRT execution context");
        Ok(TrtExecutionContext {
            context_handle: ctx_handle,
            engine: self,
        })
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nordocr_trt_sys::trt_destroy_engine(self.handle) };
        }
    }
}

/// An execution context bound to an engine for running inference.
pub struct TrtExecutionContext {
    context_handle: *mut c_void,
    engine: TrtEngine,
}

unsafe impl Send for TrtExecutionContext {}

impl TrtExecutionContext {
    /// Set the input tensor shape (for dynamic shapes).
    pub fn set_input_shape(&mut self, name: &str, dims: &[i64]) -> Result<()> {
        let name_c = CString::new(name)
            .map_err(|_| OcrError::InvalidInput("invalid tensor name".into()))?;
        let ok = unsafe {
            nordocr_trt_sys::trt_context_set_input_shape(
                self.context_handle,
                name_c.as_ptr(),
                dims.as_ptr(),
                dims.len() as i32,
            )
        };
        if ok == 0 {
            return Err(OcrError::TensorRt(format!(
                "failed to set input shape for '{name}': {dims:?}"
            )));
        }
        tracing::trace!(tensor = name, ?dims, "set input shape");
        Ok(())
    }

    /// Bind a GPU buffer to a named I/O tensor.
    pub fn set_tensor_address<T: bytemuck::Pod>(
        &mut self,
        name: &str,
        buffer: &GpuBuffer<T>,
    ) -> Result<()> {
        self.set_tensor_address_raw(name, buffer.device_ptr())
    }

    /// Bind a raw device pointer to a named I/O tensor.
    pub fn set_tensor_address_raw(&mut self, name: &str, ptr: u64) -> Result<()> {
        let name_c = CString::new(name)
            .map_err(|_| OcrError::InvalidInput("invalid tensor name".into()))?;
        let ok = unsafe {
            nordocr_trt_sys::trt_context_set_tensor_address(
                self.context_handle,
                name_c.as_ptr(),
                ptr as *mut c_void,
            )
        };
        if ok == 0 {
            return Err(OcrError::TensorRt(format!(
                "failed to bind tensor address for '{name}'"
            )));
        }
        tracing::trace!(tensor = name, ptr, "bound tensor address");
        Ok(())
    }

    /// Enqueue inference on the given CUDA stream.
    pub fn enqueue_v3(&self, stream: u64) -> Result<()> {
        let ok = unsafe {
            nordocr_trt_sys::trt_context_enqueue_v3(self.context_handle, stream)
        };
        if ok == 0 {
            return Err(OcrError::TensorRt("enqueueV3 failed".into()));
        }
        tracing::trace!(stream, "enqueued TensorRT inference");
        Ok(())
    }

    pub fn engine(&self) -> &TrtEngine {
        &self.engine
    }
}

impl Drop for TrtExecutionContext {
    fn drop(&mut self) {
        if !self.context_handle.is_null() {
            unsafe { nordocr_trt_sys::trt_destroy_context(self.context_handle) };
        }
    }
}
