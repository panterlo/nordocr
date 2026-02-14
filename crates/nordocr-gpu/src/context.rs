use std::sync::Arc;

use cudarc::driver::CudaDevice;

use nordocr_core::{OcrError, Result};

use crate::memory::GpuMemoryPool;
use crate::stream::StreamPool;

/// Default GPU memory pool size: 256 MB.
const DEFAULT_POOL_SIZE: usize = 256 * 1024 * 1024;

/// Default number of CUDA streams in the pool.
const DEFAULT_STREAM_COUNT: usize = 4;

/// Central GPU context holding device handle, memory pool, and stream pool.
///
/// Created once at startup and shared across all pipeline stages.
pub struct GpuContext {
    pub device: Arc<CudaDevice>,
    pub memory_pool: GpuMemoryPool,
    pub stream_pool: StreamPool,
    device_ordinal: usize,
}

/// Configuration for GPU context initialization.
pub struct GpuContextConfig {
    /// GPU device ordinal (default 0).
    pub device_ordinal: usize,
    /// Initial memory pool size in bytes.
    pub pool_size: usize,
    /// Number of CUDA streams to create.
    pub stream_count: usize,
}

impl Default for GpuContextConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            pool_size: DEFAULT_POOL_SIZE,
            stream_count: DEFAULT_STREAM_COUNT,
        }
    }
}

impl GpuContext {
    /// Initialize the GPU context with the given configuration.
    pub fn new(config: GpuContextConfig) -> Result<Self> {
        tracing::info!(
            device = config.device_ordinal,
            pool_mb = config.pool_size / (1024 * 1024),
            streams = config.stream_count,
            "initializing GPU context"
        );

        let device = CudaDevice::new(config.device_ordinal)
            .map_err(|e| OcrError::Cuda(format!("device init failed: {e}")))?;

        let memory_pool = GpuMemoryPool::new(device.clone(), config.pool_size)?;
        let stream_pool = StreamPool::new(device.clone(), config.stream_count)?;

        Ok(Self {
            device,
            memory_pool,
            stream_pool,
            device_ordinal: config.device_ordinal,
        })
    }

    /// Initialize with default settings on GPU 0.
    pub fn default_device() -> Result<Self> {
        Self::new(GpuContextConfig::default())
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Synchronize the device (wait for all GPU work to complete).
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| OcrError::Cuda(format!("device sync failed: {e}")))
    }
}
