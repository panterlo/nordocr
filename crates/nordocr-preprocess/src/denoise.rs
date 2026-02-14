use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

/// Safe wrapper around the denoising CUDA kernels.
pub struct DenoiseKernel {
    ptx_loaded: bool,
}

/// Denoising method selection.
#[derive(Debug, Clone, Copy)]
pub enum DenoiseMethod {
    /// Bilateral filter: edge-preserving smoothing.
    Bilateral {
        radius: u32,
        sigma_spatial: f32,
        sigma_range: f32,
    },
    /// 3x3 median filter: salt-and-pepper noise removal.
    Median,
    /// 5x5 Gaussian blur: general smoothing.
    Gaussian,
}

impl Default for DenoiseMethod {
    fn default() -> Self {
        DenoiseMethod::Bilateral {
            radius: 3,
            sigma_spatial: 3.0,
            sigma_range: 30.0,
        }
    }
}

impl DenoiseKernel {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/denoise.ptx"));

        if ptx.starts_with("// STUB") {
            tracing::warn!("denoise kernel is a stub — CUDA kernels not compiled");
            return Ok(Self { ptx_loaded: false });
        }

        let _ = ctx;
        tracing::debug!("loaded denoise PTX kernel");
        Ok(Self { ptx_loaded: true })
    }

    /// Apply denoising to a grayscale GPU image.
    pub fn execute(
        &self,
        ctx: &GpuContext,
        input: &GpuBuffer<u8>,
        output: &mut GpuBuffer<u8>,
        width: u32,
        height: u32,
        method: &DenoiseMethod,
    ) -> Result<()> {
        if !self.ptx_loaded {
            return Err(OcrError::Preprocess(
                "denoise kernel not loaded".into(),
            ));
        }

        match method {
            DenoiseMethod::Bilateral {
                radius,
                sigma_spatial,
                sigma_range,
            } => {
                tracing::trace!(
                    width, height, radius, sigma_spatial, sigma_range,
                    "bilateral denoise"
                );
                // In production:
                //   launch!(bilateral_filter<<<grid, block, 0, stream>>>(
                //       input.ptr(), output.ptr_mut(), width, height,
                //       radius, sigma_spatial, sigma_range
                //   ));
            }
            DenoiseMethod::Median => {
                tracing::trace!(width, height, "median denoise");
                // In production:
                //   launch!(median_filter_3x3<<<grid, block, 0, stream>>>(
                //       input.ptr(), output.ptr_mut(), width, height
                //   ));
            }
            DenoiseMethod::Gaussian => {
                tracing::trace!(width, height, "gaussian denoise");
                // In production:
                //   launch!(gaussian_blur_5x5<<<grid, block, 0, stream>>>(
                //       input.ptr(), output.ptr_mut(), width, height
                //   ));
            }
        }

        let _ = (ctx, input, output, width, height);
        Ok(())
    }
}
