use cudarc::driver::{CudaDevice, CudaStream as CudarcStream};
use parking_lot::Mutex;
use std::sync::Arc;

use nordocr_core::Result;

/// A pool of CUDA streams for overlapping work across pipeline stages and pages.
pub struct StreamPool {
    device: Arc<CudaDevice>,
    streams: Mutex<Vec<CudaStreamHandle>>,
    pool_size: usize,
}

/// Handle to a CUDA stream borrowed from the pool.
pub struct CudaStreamHandle {
    stream: CudarcStream,
    index: usize,
}

impl CudaStreamHandle {
    pub fn stream(&self) -> &CudarcStream {
        &self.stream
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

impl StreamPool {
    /// Create a pool of `count` CUDA streams.
    pub fn new(device: Arc<CudaDevice>, count: usize) -> Result<Self> {
        let mut streams = Vec::with_capacity(count);
        for i in 0..count {
            let stream = device
                .fork_default_stream()
                .map_err(|e| nordocr_core::OcrError::Cuda(format!("stream creation failed: {e}")))?;
            streams.push(CudaStreamHandle {
                stream,
                index: i,
            });
        }

        Ok(Self {
            device,
            streams: Mutex::new(streams),
            pool_size: count,
        })
    }

    /// Borrow a stream from the pool. Returns `None` if all streams are in use.
    pub fn try_acquire(&self) -> Option<CudaStreamHandle> {
        self.streams.lock().pop()
    }

    /// Return a stream to the pool after use.
    pub fn release(&self, handle: CudaStreamHandle) {
        self.streams.lock().push(handle);
    }

    /// Number of streams in this pool.
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Number of streams currently available.
    pub fn available(&self) -> usize {
        self.streams.lock().len()
    }

    /// Synchronize all streams in the pool (wait for all GPU work to complete).
    pub fn sync_all(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| nordocr_core::OcrError::Cuda(format!("sync failed: {e}")))
    }
}
