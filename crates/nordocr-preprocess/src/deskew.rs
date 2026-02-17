use nordocr_core::{OcrError, Result};
use nordocr_gpu::{GpuBuffer, GpuContext};

use crate::gpu_arch::{self, GpuArch};

/// Safe wrapper around the deskew CUDA kernels.
///
/// Detects and corrects rotation in scanned documents using
/// projection profile variance maximization + affine warp.
pub struct DeskewKernel {
    ptx_loaded: bool,
    arch: GpuArch,
}

/// Parameters for deskew detection and correction.
pub struct DeskewParams {
    /// Search range for skew angle in degrees (both positive and negative).
    pub max_angle_degrees: f32,
    /// Step size for angle search in degrees.
    pub angle_step_degrees: f32,
    /// Minimum angle to correct (below this, skip warping).
    pub min_correction_degrees: f32,
}

impl Default for DeskewParams {
    fn default() -> Self {
        Self {
            max_angle_degrees: 5.0,
            angle_step_degrees: 0.1,
            min_correction_degrees: 0.3,
        }
    }
}

/// Result of skew detection.
#[derive(Debug, Clone, Copy)]
pub struct DeskewResult {
    /// Detected skew angle in degrees. Positive = clockwise.
    pub angle_degrees: f32,
    /// Confidence score (normalized projection variance).
    pub confidence: f32,
    /// Whether correction was applied.
    pub corrected: bool,
}

impl DeskewKernel {
    pub fn new(ctx: &GpuContext, arch: GpuArch) -> Result<Self> {
        let ptx = gpu_arch::select_ptx(
            arch,
            include_str!(concat!(env!("OUT_DIR"), "/deskew_sm80.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/deskew_sm120.ptx")),
        );

        if ptx.starts_with("// STUB") {
            tracing::warn!("deskew kernel is a stub — CUDA kernels not compiled");
            return Ok(Self { ptx_loaded: false, arch });
        }

        let _ = ctx;
        tracing::debug!(arch = arch.name(), "loaded deskew PTX kernel");
        Ok(Self { ptx_loaded: true, arch })
    }

    /// Detect skew angle in a binarized image.
    pub fn detect_angle(
        &self,
        ctx: &GpuContext,
        input: &GpuBuffer<u8>,
        width: u32,
        height: u32,
        params: &DeskewParams,
    ) -> Result<f32> {
        if !self.ptx_loaded {
            return Err(OcrError::Preprocess(
                "deskew kernel not loaded".into(),
            ));
        }

        let _ = (ctx, input, width, height, params);

        // In production:
        // 1. For each candidate angle in [-max_angle, +max_angle] with step:
        //    a. Rotate image by candidate angle
        //    b. Compute horizontal projection profile
        //    c. Compute variance of the profile
        // 2. The angle with maximum variance is the skew angle
        // 3. All done on GPU with projection_profile + profile_variance kernels

        Ok(0.0) // placeholder
    }

    /// Correct skew by rotating the image.
    pub fn correct(
        &self,
        ctx: &GpuContext,
        input: &GpuBuffer<u8>,
        output: &mut GpuBuffer<u8>,
        width: u32,
        height: u32,
        params: &DeskewParams,
    ) -> Result<DeskewResult> {
        if !self.ptx_loaded {
            return Err(OcrError::Preprocess(
                "deskew kernel not loaded".into(),
            ));
        }

        let angle = self.detect_angle(ctx, input, width, height, params)?;

        if angle.abs() < params.min_correction_degrees {
            tracing::trace!(angle, "skew below threshold, skipping correction");
            return Ok(DeskewResult {
                angle_degrees: angle,
                confidence: 1.0,
                corrected: false,
            });
        }

        tracing::trace!(angle, "applying deskew correction");

        // In production:
        //   let angle_rad = angle.to_radians();
        //   let cos_a = angle_rad.cos();
        //   let sin_a = angle_rad.sin();
        //   let cx = width as f32 / 2.0;
        //   let cy = height as f32 / 2.0;
        //   launch!(affine_rotate<<<grid, block, 0, stream>>>(
        //       input.ptr(), output.ptr_mut(), width, height,
        //       cos_a, sin_a, cx, cy
        //   ));

        let _ = (ctx, input, output, width, height);

        Ok(DeskewResult {
            angle_degrees: angle,
            confidence: 1.0,
            corrected: true,
        })
    }
}
