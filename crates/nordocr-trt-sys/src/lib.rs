//! Raw FFI bindings to the TensorRT C shim (`trt_shim.cpp`).
//!
//! The C++ shim wraps the TensorRT 10.x C++ API as `extern "C"` functions,
//! compiled by the `cc` crate at build time and linked against `nvinfer_10`.
//!
//! # Safety
//! All functions in this module are unsafe FFI calls. Use `nordocr-trt`
//! for safe Rust wrappers.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// --- Feature: bindgen-generated bindings from full TRT headers ---
#[cfg(feature = "generated")]
include!(concat!(env!("OUT_DIR"), "/trt_bindings.rs"));

// --- Default: hand-written FFI matching trt_shim.cpp ---
#[cfg(not(feature = "generated"))]
pub mod ffi {
    use std::os::raw::c_char;

    extern "C" {
        // Runtime lifecycle
        pub fn trt_create_runtime() -> *mut std::ffi::c_void;
        pub fn trt_set_log_severity(severity: i32);
        pub fn trt_destroy_runtime(runtime: *mut std::ffi::c_void);

        // Engine lifecycle
        pub fn trt_deserialize_engine(
            runtime: *mut std::ffi::c_void,
            data: *const std::ffi::c_void,
            size: u64,
        ) -> *mut std::ffi::c_void;
        pub fn trt_destroy_engine(engine: *mut std::ffi::c_void);

        // Engine introspection
        pub fn trt_engine_get_nb_io_tensors(engine: *mut std::ffi::c_void) -> i32;
        pub fn trt_engine_get_tensor_name(
            engine: *mut std::ffi::c_void,
            index: i32,
        ) -> *const c_char;
        pub fn trt_engine_get_tensor_io_mode(
            engine: *mut std::ffi::c_void,
            name: *const c_char,
        ) -> i32;
        pub fn trt_engine_get_tensor_shape(
            engine: *mut std::ffi::c_void,
            name: *const c_char,
            out_dims: *mut i64,
            max_dims: i32,
        ) -> i32;
        pub fn trt_engine_get_tensor_dtype(
            engine: *mut std::ffi::c_void,
            name: *const c_char,
        ) -> i32;

        // Execution context
        pub fn trt_create_execution_context(
            engine: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void;
        pub fn trt_destroy_context(context: *mut std::ffi::c_void);
        pub fn trt_context_set_input_shape(
            context: *mut std::ffi::c_void,
            name: *const c_char,
            dims: *const i64,
            nb_dims: i32,
        ) -> i32;
        pub fn trt_context_set_tensor_address(
            context: *mut std::ffi::c_void,
            name: *const c_char,
            gpu_ptr: *mut std::ffi::c_void,
        ) -> i32;
        pub fn trt_context_enqueue_v3(context: *mut std::ffi::c_void, stream: u64) -> i32;

        // Version info
        pub fn trt_get_version_major() -> i32;
        pub fn trt_get_version_minor() -> i32;
        pub fn trt_get_version_patch() -> i32;
        pub fn trt_get_version_build() -> i32;
    }
}

#[cfg(not(feature = "generated"))]
pub use ffi::*;

/// TensorRT data types (matches `nvinfer1::DataType` enum values).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrtDtype {
    Float = 0,
    Half = 1,
    Int8 = 2,
    Int32 = 3,
    Bool = 4,
    Uint8 = 5,
    Fp8E4M3 = 6,
    Bf16 = 7,
    Int64 = 8,
    Int4 = 9,
    Fp4 = 10,
}

impl TrtDtype {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Float),
            1 => Some(Self::Half),
            2 => Some(Self::Int8),
            3 => Some(Self::Int32),
            4 => Some(Self::Bool),
            5 => Some(Self::Uint8),
            6 => Some(Self::Fp8E4M3),
            7 => Some(Self::Bf16),
            8 => Some(Self::Int64),
            9 => Some(Self::Int4),
            10 => Some(Self::Fp4),
            _ => None,
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Self::Float | Self::Int32 => 4,
            Self::Half | Self::Bf16 => 2,
            Self::Int8 | Self::Uint8 | Self::Fp8E4M3 | Self::Bool => 1,
            Self::Int64 => 8,
            Self::Int4 | Self::Fp4 => 1, // packed, minimum addressable
        }
    }
}

/// TensorRT tensor I/O mode (matches `nvinfer1::TensorIOMode`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrtTensorIOMode {
    None = 0,
    Input = 1,
    Output = 2,
}

impl TrtTensorIOMode {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Input),
            2 => Some(Self::Output),
            _ => None,
        }
    }
}
