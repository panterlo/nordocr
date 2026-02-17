use nordocr_core::Result;
use nordocr_core::TextRegion;
use nordocr_gpu::GpuContext;
use nordocr_preprocess::gpu_arch::GpuArch;
use nordocr_preprocess::{DetectPreprocessKernel, DetectPreprocessParams};

use crate::contour;
use crate::morphological::MorphologicalDetector;

/// GPU-accelerated morphological text region detector.
///
/// Runs image processing (grayscale, blur, threshold, dilation) on GPU,
/// then performs CCL and filtering on CPU. Produces identical results to
/// `MorphologicalDetector` but ~6.5x faster by offloading pixel processing.
pub struct GpuMorphologicalDetector {
    preprocess: DetectPreprocessKernel,
    /// CPU-side config reused for filtering, clustering, and as param source.
    config: MorphologicalDetector,
}

impl GpuMorphologicalDetector {
    /// Initialize the GPU detector, loading CUDA kernels for the detected architecture.
    ///
    /// Returns `Err` if the detect_preprocess PTX is not available (stub-only build).
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let arch = GpuArch::detect(ctx)?;
        let preprocess = DetectPreprocessKernel::new(ctx, arch)?;

        tracing::info!("GPU morphological detector initialized");

        Ok(Self {
            preprocess,
            config: MorphologicalDetector::default(),
        })
    }

    /// Initialize with explicit configuration (overrides default params).
    pub fn with_config(ctx: &GpuContext, config: MorphologicalDetector) -> Result<Self> {
        let arch = GpuArch::detect(ctx)?;
        let preprocess = DetectPreprocessKernel::new(ctx, arch)?;

        tracing::info!("GPU morphological detector initialized (custom config)");

        Ok(Self { preprocess, config })
    }

    /// Detect text regions in an RGB u8 HWC image.
    ///
    /// GPU: RGB → grayscale → blur → threshold → dilation → binary mask download.
    /// CPU: CCL → bbox extraction → size/margin/rotation filtering → row/column clustering.
    pub fn detect(
        &self,
        ctx: &GpuContext,
        rgb: &[u8],
        w: u32,
        h: u32,
        page_index: u32,
    ) -> Result<Vec<TextRegion>> {
        let params = DetectPreprocessParams {
            blur_size: self.config.blur_size,
            adaptive_block_size: self.config.adaptive_block_size,
            adaptive_c: self.config.adaptive_c as f32,
            dilate_kernel_w: self.config.kernel_w,
            dilate_kernel_h: self.config.kernel_h,
            dilate_iterations: self.config.iterations,
        };

        // GPU preprocessing → binary masks (CPU memory).
        let (dilated, pre_dilation) = self.preprocess.execute(ctx, rgb, w, h, &params)?;

        // CPU CCL → components with bounding boxes and orientation.
        let components = contour::extract_components(&dilated, w, h);

        // Split multi-line merges using horizontal projection profile.
        let components =
            contour::split_tall_components(components, &dilated, w, h, self.config.max_height_ratio);

        // Trim small edge fragments (removes leading/trailing garbage).
        let components = contour::trim_edge_fragments(components, &pre_dilation, w, h);

        tracing::debug!(
            page = page_index,
            raw_regions = components.len(),
            "GPU morphological detection: raw components"
        );

        // CPU filtering + clustering (reuse MorphologicalDetector logic).
        let regions = self.config.filter_and_cluster(components, w, h, page_index);

        Ok(regions)
    }
}
