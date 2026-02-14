use half::f16;
use nordocr_core::{BBox, OcrError, Polygon, Result, TextRegion};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Post-processes DBNet++ probability maps into text region bounding boxes.
///
/// Pipeline (all on GPU):
/// 1. Threshold the probability map to get a binary mask
/// 2. Find contours / connected components
/// 3. Fit minimum bounding polygons
/// 4. Convert to axis-aligned bounding boxes with optional polygon output
pub struct DetectionPostprocessor {
    /// Probability threshold for text/non-text classification.
    prob_threshold: f32,
    /// Minimum area (in pixels) for a text region to be kept.
    min_area: f32,
    /// Maximum number of text regions to return per page.
    max_regions: usize,
    /// How much to expand detected boxes (accounts for DBNet shrinkage).
    expand_ratio: f32,
}

impl DetectionPostprocessor {
    pub fn new() -> Self {
        Self {
            prob_threshold: 0.3,
            min_area: 100.0,
            max_regions: 1000,
            expand_ratio: 1.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.prob_threshold = threshold;
        self
    }

    pub fn with_min_area(mut self, area: f32) -> Self {
        self.min_area = area;
        self
    }

    pub fn with_expand_ratio(mut self, ratio: f32) -> Self {
        self.expand_ratio = ratio;
        self
    }

    /// Extract text regions from the probability map.
    ///
    /// In production, steps 1-2 run on GPU via custom CUDA kernels,
    /// and step 3-4 run on CPU (contour fitting is irregular and
    /// poorly suited to GPU). The probability map is the only data
    /// transferred to CPU in this step.
    pub fn extract_regions(
        &self,
        ctx: &GpuContext,
        prob_map: &GpuBuffer<f16>,
        width: u32,
        height: u32,
        page_index: u32,
    ) -> Result<Vec<TextRegion>> {
        let pixel_count = (width * height) as usize;

        // Step 1: Threshold on GPU → binary mask.
        // In production:
        //   let mask = ctx.memory_pool.alloc::<u8>(pixel_count)?;
        //   launch!(threshold_kernel<<<grid, block>>>(
        //       prob_map.ptr(), mask.ptr_mut(), pixel_count, self.prob_threshold
        //   ));

        // Step 2: Connected component labeling on GPU.
        // In production:
        //   launch!(ccl_kernel<<<...>>>(mask, labels, width, height));

        // Step 3: Copy contour data to CPU for polygon fitting.
        // This is the only CPU readback in the detection pipeline.
        // In production:
        //   let labels_cpu = ctx.device.dtoh_sync_copy(&labels)?;
        //   let contours = find_contours(&labels_cpu, width, height);

        // Step 4: Fit bounding boxes and filter.
        // Placeholder: return empty until models are connected.
        let _ = (ctx, prob_map, width, height, pixel_count);

        tracing::trace!(
            page = page_index,
            threshold = self.prob_threshold,
            "extracting text regions"
        );

        Ok(Vec::new())
    }

    /// Expand a bounding box by the configured ratio (compensates for
    /// DBNet's Vatti shrinkage during training).
    fn expand_box(&self, bbox: &BBox) -> BBox {
        let dx = bbox.width * (self.expand_ratio - 1.0) / 2.0;
        let dy = bbox.height * (self.expand_ratio - 1.0) / 2.0;
        BBox::new(
            (bbox.x - dx).max(0.0),
            (bbox.y - dy).max(0.0),
            bbox.width + 2.0 * dx,
            bbox.height + 2.0 * dy,
        )
    }
}

impl Default for DetectionPostprocessor {
    fn default() -> Self {
        Self::new()
    }
}
