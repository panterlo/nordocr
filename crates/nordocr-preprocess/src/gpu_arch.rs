use nordocr_core::{OcrError, Result};
use nordocr_gpu::GpuContext;

/// Supported GPU architectures for kernel loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArch {
    /// NVIDIA A6000 Ada Lovelace — sm_89 (development)
    AdaSm89,
    /// NVIDIA RTX 6000 PRO Blackwell — sm_120 (production)
    BlackwellSm120,
}

impl GpuArch {
    /// Detect the GPU architecture from the CUDA device.
    pub fn detect(ctx: &GpuContext) -> Result<Self> {
        // In production with cudarc:
        //   let major = ctx.device.attribute(CudaDeviceAttribute::ComputeCapabilityMajor)?;
        //   let minor = ctx.device.attribute(CudaDeviceAttribute::ComputeCapabilityMinor)?;
        //   let sm = major * 10 + minor;
        //
        // For now, check the NORDOCR_TARGET_ARCH env var as fallback.
        let _ = ctx;

        if let Ok(arch) = std::env::var("NORDOCR_TARGET_ARCH") {
            return match arch.as_str() {
                "sm_89" | "89" => Ok(GpuArch::AdaSm89),
                "sm_120" | "120" => Ok(GpuArch::BlackwellSm120),
                other => Err(OcrError::Cuda(format!(
                    "unknown target arch '{other}', expected sm_89 or sm_120"
                ))),
            };
        }

        // Default to sm_89 (dev) — in production this is replaced by runtime detection.
        tracing::warn!("GPU arch not detected, defaulting to sm_89 (Ada)");
        Ok(GpuArch::AdaSm89)
    }

    pub fn sm_version(&self) -> u32 {
        match self {
            GpuArch::AdaSm89 => 89,
            GpuArch::BlackwellSm120 => 120,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            GpuArch::AdaSm89 => "Ada Lovelace (sm_89)",
            GpuArch::BlackwellSm120 => "Blackwell (sm_120)",
        }
    }

    /// Suffix used in compiled PTX filenames (matches build.rs output).
    pub fn ptx_suffix(&self) -> &'static str {
        match self {
            GpuArch::AdaSm89 => "sm89",
            GpuArch::BlackwellSm120 => "sm120",
        }
    }
}

/// Select the correct PTX source for the detected GPU architecture.
///
/// At build time, nvcc compiles each .cu file to multiple PTX variants:
///   {kernel}_sm89.ptx   — for A6000 Ada (dev)
///   {kernel}_sm120.ptx  — for RTX 6000 PRO Blackwell (prod)
///
/// This function picks the right one at runtime based on the detected GPU.
/// If the exact match isn't available, it falls back to the lower arch
/// (CUDA driver can JIT-compile upward from older PTX).
pub fn select_ptx(arch: GpuArch, sm89_ptx: &'static str, sm120_ptx: &'static str) -> &'static str {
    match arch {
        GpuArch::BlackwellSm120 => {
            if !sm120_ptx.starts_with("// STUB") {
                return sm120_ptx;
            }
            tracing::warn!("sm_120 PTX not available, falling back to sm_89");
            sm89_ptx
        }
        GpuArch::AdaSm89 => sm89_ptx,
    }
}
