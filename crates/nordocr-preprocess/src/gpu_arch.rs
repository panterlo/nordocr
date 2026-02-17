use nordocr_core::{OcrError, Result};
use nordocr_gpu::GpuContext;

/// Supported GPU architectures for kernel loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArch {
    /// NVIDIA Ampere (sm_80) — covers A6000 (sm_86) and Ada (sm_89) via JIT
    AmpereSm80,
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
                "sm_80" | "80" | "sm_86" | "86" | "sm_89" | "89" => Ok(GpuArch::AmpereSm80),
                "sm_120" | "120" => Ok(GpuArch::BlackwellSm120),
                other => Err(OcrError::Cuda(format!(
                    "unknown target arch '{other}', expected sm_80/sm_86/sm_89 or sm_120"
                ))),
            };
        }

        // Default to sm_80 (dev) — in production this is replaced by runtime detection.
        tracing::warn!("GPU arch not detected, defaulting to sm_80 (Ampere)");
        Ok(GpuArch::AmpereSm80)
    }

    pub fn sm_version(&self) -> u32 {
        match self {
            GpuArch::AmpereSm80 => 80,
            GpuArch::BlackwellSm120 => 120,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            GpuArch::AmpereSm80 => "Ampere (sm_80)",
            GpuArch::BlackwellSm120 => "Blackwell (sm_120)",
        }
    }

    /// Suffix used in compiled PTX filenames (matches build.rs output).
    pub fn ptx_suffix(&self) -> &'static str {
        match self {
            GpuArch::AmpereSm80 => "sm80",
            GpuArch::BlackwellSm120 => "sm120",
        }
    }
}

/// Select the correct PTX source for the detected GPU architecture.
///
/// At build time, nvcc compiles each .cu file to multiple PTX variants:
///   {kernel}_sm80.ptx   — for Ampere/Ada (dev)
///   {kernel}_sm120.ptx  — for RTX 6000 PRO Blackwell (prod)
///
/// This function picks the right one at runtime based on the detected GPU.
/// If the exact match isn't available, it falls back to the lower arch
/// (CUDA driver can JIT-compile upward from older PTX).
pub fn select_ptx(arch: GpuArch, sm80_ptx: &'static str, sm120_ptx: &'static str) -> &'static str {
    match arch {
        GpuArch::BlackwellSm120 => {
            if !sm120_ptx.starts_with("// STUB") {
                return sm120_ptx;
            }
            tracing::warn!("sm_120 PTX not available, falling back to sm_80");
            sm80_ptx
        }
        GpuArch::AmpereSm80 => sm80_ptx,
    }
}
