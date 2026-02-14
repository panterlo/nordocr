use nordocr_core::Result;
use nordocr_gpu::{CudaStreamHandle, GpuContext};

/// Schedules pages across CUDA streams for overlapped processing.
///
/// When multiple pages arrive, they can be processed in parallel on
/// different CUDA streams. The scheduler assigns pages to available
/// streams and manages synchronization.
pub struct PageScheduler {
    max_concurrent: usize,
}

/// A scheduled unit of work: one or more pages assigned to a stream.
#[derive(Debug)]
pub struct ScheduledWork {
    pub page_indices: Vec<usize>,
    pub stream_index: usize,
}

impl PageScheduler {
    pub fn new(num_streams: usize) -> Self {
        Self {
            max_concurrent: num_streams,
        }
    }

    /// Create a work schedule for the given number of pages.
    ///
    /// Distributes pages round-robin across available streams.
    /// If there are more pages than streams, pages are batched
    /// within each stream.
    pub fn schedule(&self, num_pages: usize) -> Vec<ScheduledWork> {
        let mut work = Vec::new();

        // Distribute pages across streams round-robin.
        let mut stream_pages: Vec<Vec<usize>> = vec![Vec::new(); self.max_concurrent];

        for page_idx in 0..num_pages {
            let stream_idx = page_idx % self.max_concurrent;
            stream_pages[stream_idx].push(page_idx);
        }

        for (stream_index, pages) in stream_pages.into_iter().enumerate() {
            if !pages.is_empty() {
                work.push(ScheduledWork {
                    page_indices: pages,
                    stream_index,
                });
            }
        }

        tracing::debug!(
            pages = num_pages,
            streams = work.len(),
            "scheduled page processing"
        );

        work
    }

    /// Acquire a CUDA stream from the pool for a scheduled work unit.
    pub fn acquire_stream(&self, ctx: &GpuContext) -> Result<Option<CudaStreamHandle>> {
        Ok(ctx.stream_pool.try_acquire())
    }

    /// Release a stream back to the pool.
    pub fn release_stream(&self, ctx: &GpuContext, stream: CudaStreamHandle) {
        ctx.stream_pool.release(stream);
    }
}
