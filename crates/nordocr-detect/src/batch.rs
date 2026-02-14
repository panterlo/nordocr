use half::f16;
use nordocr_core::{Result, TextRegion};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::engine::DetectionEngine;
use crate::postprocess::DetectionPostprocessor;

/// Accumulates pages into batches for efficient detection inference.
///
/// Since detection processes full pages, batching is straightforward:
/// group pages into batches of `max_batch_size` and run inference.
pub struct DetectionBatcher {
    engine: DetectionEngine,
    postprocessor: DetectionPostprocessor,
}

impl DetectionBatcher {
    pub fn new(engine: DetectionEngine, postprocessor: DetectionPostprocessor) -> Self {
        Self {
            engine,
            postprocessor,
        }
    }

    /// Detect text regions across multiple pages.
    ///
    /// Pages are batched to maximize GPU utilization. Returns text
    /// regions grouped by page index.
    pub fn detect_pages(
        &mut self,
        ctx: &GpuContext,
        pages: &[(GpuBuffer<f16>, u32, u32)], // (resized_image, width, height)
        stream: u64,
    ) -> Result<Vec<Vec<TextRegion>>> {
        let max_batch = self.engine.max_batch_size() as usize;
        let mut all_regions = Vec::with_capacity(pages.len());

        for chunk in pages.chunks(max_batch) {
            let batch_size = chunk.len() as u32;

            // In production:
            // 1. Stack page images into a single batched tensor on GPU.
            // 2. Run batched inference.
            // 3. Split output probability maps by page.

            // For each page in the batch, we need a reference to its preprocessed buffer.
            // Using the first page's buffer as placeholder.
            if let Some((first_buf, _, _)) = chunk.first() {
                let prob_map = self.engine.infer(first_buf, batch_size, stream)?;

                // Post-process each page's probability map.
                for (i, (_, width, height)) in chunk.iter().enumerate() {
                    let page_index = all_regions.len() as u32;
                    let regions = self.postprocessor.extract_regions(
                        ctx,
                        prob_map,
                        *width,
                        *height,
                        page_index,
                    )?;
                    all_regions.push(regions);
                }
            }
        }

        Ok(all_regions)
    }

    pub fn engine(&self) -> &DetectionEngine {
        &self.engine
    }
}
