use crate::error::Result;

/// A GPU buffer handle that pipeline stages pass between each other.
/// The actual GPU memory is managed by `nordocr-gpu`; this is the
/// type-erased token used at the trait boundary.
#[derive(Debug)]
pub struct GpuBufferHandle {
    /// Device pointer (as usize for FFI-safety across crate boundaries).
    pub ptr: usize,
    /// Size in bytes.
    pub size: usize,
    /// Owning pool ID, for deallocation routing.
    pub pool_id: u64,
}

/// Trait for a single stage in the OCR pipeline.
///
/// Each stage operates entirely on GPU memory — receiving a GPU buffer
/// and producing a GPU buffer, with no CPU round-trips in between.
pub trait PipelineStage: Send + Sync {
    /// Human-readable name for tracing/metrics.
    fn name(&self) -> &str;

    /// Execute this stage on the given GPU input, producing GPU output.
    ///
    /// `stream` is the CUDA stream ordinal to schedule work on.
    fn execute(&self, input: &GpuBufferHandle, stream: u32) -> Result<GpuBufferHandle>;
}
