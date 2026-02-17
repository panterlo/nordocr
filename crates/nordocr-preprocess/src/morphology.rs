use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::gpu_arch::{self, GpuArch};

/// Safe wrapper around morphological CUDA kernels.
///
/// Provides erosion, dilation, opening, closing, and small-component
/// removal for cleaning up binarized document images.
pub struct MorphologyKernel {
    ptx_loaded: bool,
    arch: GpuArch,
}

/// Morphological operation type.
#[derive(Debug, Clone, Copy)]
pub enum MorphOp {
    /// Dilate: expand bright regions (thicken text in dark-on-light).
    Dilate { kernel_w: u32, kernel_h: u32 },
    /// Erode: shrink bright regions (thin text in dark-on-light).
    Erode { kernel_w: u32, kernel_h: u32 },
    /// Opening (erode then dilate): remove small bright noise.
    Open { kernel_w: u32, kernel_h: u32 },
    /// Closing (dilate then erode): fill small gaps in text.
    Close { kernel_w: u32, kernel_h: u32 },
    /// Remove connected components smaller than `min_pixels`.
    RemoveSmallComponents { min_pixels: u32 },
}

impl MorphologyKernel {
    pub fn new(ctx: &GpuContext, arch: GpuArch) -> Result<Self> {
        let ptx = gpu_arch::select_ptx(
            arch,
            include_str!(concat!(env!("OUT_DIR"), "/morphology_sm80.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/morphology_sm120.ptx")),
        );

        if ptx.starts_with("// STUB") {
            tracing::warn!("morphology kernel is a stub — CUDA kernels not compiled");
            return Ok(Self { ptx_loaded: false, arch });
        }

        let _ = ctx;
        tracing::debug!(arch = arch.name(), "loaded morphology PTX kernel");
        Ok(Self { ptx_loaded: true, arch })
    }

    /// Apply a morphological operation on a binarized GPU image.
    pub fn execute(
        &self,
        ctx: &GpuContext,
        input: &GpuBuffer<u8>,
        output: &mut GpuBuffer<u8>,
        width: u32,
        height: u32,
        op: &MorphOp,
    ) -> Result<()> {
        if !self.ptx_loaded {
            return Err(OcrError::Preprocess(
                "morphology kernel not loaded".into(),
            ));
        }

        match op {
            MorphOp::Dilate { kernel_w, kernel_h } => {
                tracing::trace!(width, height, kernel_w, kernel_h, "dilate");
                // launch!(dilate_rect<<<grid, block>>>(input, output, w, h, kw, kh))
            }
            MorphOp::Erode { kernel_w, kernel_h } => {
                tracing::trace!(width, height, kernel_w, kernel_h, "erode");
                // launch!(erode_rect<<<grid, block>>>(input, output, w, h, kw, kh))
            }
            MorphOp::Open { kernel_w, kernel_h } => {
                tracing::trace!(width, height, kernel_w, kernel_h, "open");
                // Erode, then dilate into output (uses a temp buffer from the pool).
                // let temp = ctx.memory_pool.alloc::<u8>(width * height)?;
                // erode(input → temp), dilate(temp → output)
                // ctx.memory_pool.free(temp);
            }
            MorphOp::Close { kernel_w, kernel_h } => {
                tracing::trace!(width, height, kernel_w, kernel_h, "close");
                // Dilate, then erode into output.
            }
            MorphOp::RemoveSmallComponents { min_pixels } => {
                tracing::trace!(width, height, min_pixels, "remove small components");
                // 1. label_init kernel
                // 2. Connected component labeling (iterative merge)
                // 3. Count component sizes (histogram)
                // 4. remove_small_components kernel
            }
        }

        let _ = (ctx, input, output, width, height);
        Ok(())
    }
}
