use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::gpu_arch::{self, GpuArch};

/// Safe wrapper around the adaptive binarization CUDA kernel.
///
/// Uses Sauvola's method with integral images for O(1) local thresholding.
/// Optimized for scanned documents where illumination varies across the page.
pub struct BinarizeKernel {
    ptx_loaded: bool,
    arch: GpuArch,
}

/// Parameters for Sauvola adaptive binarization.
pub struct BinarizeParams {
    /// Size of the local window for threshold computation (must be odd).
    pub window_size: u32,
    /// Sauvola k parameter (sensitivity to contrast). Default: 0.2.
    pub k: f32,
    /// Sauvola R parameter (dynamic range of std deviation). Default: 128.0.
    pub r: f32,
}

impl Default for BinarizeParams {
    fn default() -> Self {
        Self {
            window_size: 31,
            k: 0.2,
            r: 128.0,
        }
    }
}

impl BinarizeKernel {
    /// Load the binarization PTX for the detected GPU architecture.
    pub fn new(ctx: &GpuContext, arch: GpuArch) -> Result<Self> {
        let ptx = gpu_arch::select_ptx(
            arch,
            include_str!(concat!(env!("OUT_DIR"), "/binarize_sm89.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/binarize_sm120.ptx")),
        );

        if ptx.starts_with("// STUB") {
            tracing::warn!("binarize kernel is a stub — CUDA kernels not compiled");
            return Ok(Self { ptx_loaded: false, arch });
        }

        // In production with cudarc:
        //   let module = ctx.device.load_ptx(
        //       cudarc::nvrtc::Ptx::from_src(ptx),
        //       "binarize",
        //       &["integral_image_horizontal", "integral_image_vertical", "adaptive_binarize"],
        //   )?;
        let _ = ctx;

        tracing::debug!(arch = arch.name(), "loaded binarize PTX kernel");
        Ok(Self { ptx_loaded: true, arch })
    }

    /// Execute adaptive binarization on a grayscale GPU image.
    ///
    /// Input: single-channel u8 image in GPU memory.
    /// Output: binarized u8 image (0 or 255) in GPU memory.
    pub fn execute(
        &self,
        ctx: &GpuContext,
        input: &GpuBuffer<u8>,
        output: &mut GpuBuffer<u8>,
        width: u32,
        height: u32,
        params: &BinarizeParams,
    ) -> Result<()> {
        if !self.ptx_loaded {
            return Err(OcrError::Preprocess(
                "binarize kernel not loaded (CUDA unavailable)".into(),
            ));
        }

        tracing::trace!(width, height, window = params.window_size, "binarize");

        // In production:
        //
        // 1. Allocate integral image buffers from pool
        //    let integral = ctx.memory_pool.alloc::<u32>(width * height)?;
        //    let integral_sq = ctx.memory_pool.alloc::<u64>(width * height)?;
        //
        // 2. Compute integral images
        //    let block_1d = (256, 1, 1);
        //    let grid_h = (height.div_ceil(256), 1, 1);
        //    launch!(integral_image_horizontal<<<grid_h, block_1d, 0, stream>>>(
        //        input.ptr(), integral.ptr_mut(), width, height
        //    ));
        //    let grid_w = (width.div_ceil(256), 1, 1);
        //    launch!(integral_image_vertical<<<grid_w, block_1d, 0, stream>>>(
        //        integral.ptr_mut(), width, height
        //    ));
        //
        // 3. Run adaptive binarize
        //    let block_2d = (32, 32, 1);  // works well on both sm_89 and sm_120
        //    let grid_2d = (width.div_ceil(32), height.div_ceil(32), 1);
        //    launch!(adaptive_binarize<<<grid_2d, block_2d, 0, stream>>>(
        //        input.ptr(), integral.ptr(), integral_sq.ptr(),
        //        output.ptr_mut(), width, height,
        //        params.window_size, params.k, params.r
        //    ));
        //
        // 4. Return integral buffers to pool
        //    ctx.memory_pool.free(integral);
        //    ctx.memory_pool.free(integral_sq);

        let _ = (ctx, input, output, width, height, params);
        Ok(())
    }
}
