pub mod binarize;
pub mod denoise;
pub mod deskew;
pub mod detect_preprocess;
pub mod gpu_arch;
pub mod morphology;

pub use binarize::{BinarizeKernel, BinarizeParams};
pub use denoise::{DenoiseKernel, DenoiseMethod};
pub use deskew::{DeskewKernel, DeskewParams, DeskewResult};
pub use detect_preprocess::{DetectPreprocessKernel, DetectPreprocessParams};
pub use gpu_arch::GpuArch;
pub use morphology::{MorphOp, MorphologyKernel};

use nordocr_core::Result;
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Full preprocessing pipeline: denoise → deskew → binarize → morphology.
///
/// All operations run on GPU without any CPU round-trips. The output
/// is a cleaned-up binarized image ready for text detection.
pub struct PreprocessPipeline {
    denoise: DenoiseKernel,
    deskew: DeskewKernel,
    binarize: BinarizeKernel,
    morphology: MorphologyKernel,
    config: PreprocessConfig,
    arch: GpuArch,
}

/// Configuration for the preprocessing pipeline.
pub struct PreprocessConfig {
    pub denoise_method: DenoiseMethod,
    pub deskew_params: DeskewParams,
    pub binarize_params: BinarizeParams,
    /// Whether to apply morphological cleanup after binarization.
    pub enable_morphology: bool,
    /// Minimum component size (pixels) to keep. 0 = no filtering.
    pub min_component_size: u32,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            denoise_method: DenoiseMethod::default(),
            deskew_params: DeskewParams::default(),
            binarize_params: BinarizeParams::default(),
            enable_morphology: true,
            min_component_size: 10,
        }
    }
}

impl PreprocessPipeline {
    /// Initialize the preprocessing pipeline, loading all CUDA kernels
    /// for the detected (or specified) GPU architecture.
    pub fn new(ctx: &GpuContext, config: PreprocessConfig) -> Result<Self> {
        let arch = GpuArch::detect(ctx)?;
        tracing::info!(arch = arch.name(), "initializing preprocessing for GPU");

        let denoise = DenoiseKernel::new(ctx, arch)?;
        let deskew = DeskewKernel::new(ctx, arch)?;
        let binarize = BinarizeKernel::new(ctx, arch)?;
        let morphology = MorphologyKernel::new(ctx, arch)?;

        Ok(Self {
            denoise,
            deskew,
            binarize,
            morphology,
            config,
            arch,
        })
    }

    /// Initialize with a specific GPU architecture (useful for testing).
    pub fn with_arch(ctx: &GpuContext, config: PreprocessConfig, arch: GpuArch) -> Result<Self> {
        tracing::info!(arch = arch.name(), "initializing preprocessing (explicit arch)");

        let denoise = DenoiseKernel::new(ctx, arch)?;
        let deskew = DeskewKernel::new(ctx, arch)?;
        let binarize = BinarizeKernel::new(ctx, arch)?;
        let morphology = MorphologyKernel::new(ctx, arch)?;

        Ok(Self {
            denoise,
            deskew,
            binarize,
            morphology,
            config,
            arch,
        })
    }

    pub fn arch(&self) -> GpuArch {
        self.arch
    }

    /// Run the full preprocessing pipeline on a grayscale image.
    ///
    /// Input: single-channel u8 image on GPU.
    /// Output: cleaned binarized u8 image on GPU.
    pub fn execute(
        &self,
        ctx: &GpuContext,
        input: &GpuBuffer<u8>,
        width: u32,
        height: u32,
    ) -> Result<GpuBuffer<u8>> {
        let pixel_count = (width * height) as usize;

        // Allocate intermediate buffers from the GPU memory pool.
        let mut denoised = ctx.memory_pool.alloc::<u8>(pixel_count)?;
        let mut deskewed = ctx.memory_pool.alloc::<u8>(pixel_count)?;
        let mut binarized = ctx.memory_pool.alloc::<u8>(pixel_count)?;

        // Stage 1: Denoise.
        self.denoise.execute(
            ctx,
            input,
            &mut denoised,
            width,
            height,
            &self.config.denoise_method,
        )?;

        // Stage 2: Deskew.
        self.deskew.correct(
            ctx,
            &denoised,
            &mut deskewed,
            width,
            height,
            &self.config.deskew_params,
        )?;

        // Stage 3: Binarize.
        self.binarize.execute(
            ctx,
            &deskewed,
            &mut binarized,
            width,
            height,
            &self.config.binarize_params,
        )?;

        // Return intermediate buffers to pool.
        ctx.memory_pool.free(denoised);
        ctx.memory_pool.free(deskewed);

        // Stage 4: Morphological cleanup (optional).
        if self.config.enable_morphology {
            let mut cleaned = ctx.memory_pool.alloc::<u8>(pixel_count)?;

            // Close small gaps in text strokes.
            self.morphology.execute(
                ctx,
                &binarized,
                &mut cleaned,
                width,
                height,
                &MorphOp::Close {
                    kernel_w: 3,
                    kernel_h: 3,
                },
            )?;

            ctx.memory_pool.free(binarized);

            // Remove small noise components.
            if self.config.min_component_size > 0 {
                let mut final_output = ctx.memory_pool.alloc::<u8>(pixel_count)?;
                self.morphology.execute(
                    ctx,
                    &cleaned,
                    &mut final_output,
                    width,
                    height,
                    &MorphOp::RemoveSmallComponents {
                        min_pixels: self.config.min_component_size,
                    },
                )?;
                ctx.memory_pool.free(cleaned);
                return Ok(final_output);
            }

            return Ok(cleaned);
        }

        Ok(binarized)
    }
}
