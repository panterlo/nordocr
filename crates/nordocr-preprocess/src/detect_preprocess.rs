use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use nordocr_core::{OcrError, Result};
use nordocr_gpu::GpuContext;
use std::sync::Arc;

use crate::gpu_arch::{self, GpuArch};

/// GPU-accelerated preprocessing for morphological text detection.
///
/// Runs the entire image-processing pipeline on GPU:
/// RGB → grayscale → Gaussian blur → integral image → adaptive threshold → dilation
///
/// The output is a binary mask (u8, 0/255) ready for CPU-side CCL.
pub struct DetectPreprocessKernel {
    rgb_to_gray: CudaFunction,
    gaussian_blur_h: CudaFunction,
    gaussian_blur_v: CudaFunction,
    integral_image_h: CudaFunction,
    integral_image_v: CudaFunction,
    adaptive_threshold_mean: CudaFunction,
    dilate_h: CudaFunction,
    dilate_v: CudaFunction,
    _module: Arc<CudaModule>,
}

/// Parameters for GPU detection preprocessing.
///
/// Defaults match `MorphologicalDetector::default()` in nordocr-detect.
pub struct DetectPreprocessParams {
    /// Gaussian blur kernel size (must be odd, 0 = disabled). Default: 5.
    pub blur_size: u32,
    /// Block size for adaptive thresholding (must be odd). Default: 15.
    pub adaptive_block_size: u32,
    /// Constant subtracted from local mean. Default: 4.0.
    pub adaptive_c: f32,
    /// Dilation kernel width (connects characters horizontally). Default: 20.
    pub dilate_kernel_w: u32,
    /// Dilation kernel height (keeps text lines separate). Default: 3.
    pub dilate_kernel_h: u32,
    /// Number of dilation iterations. Default: 2.
    pub dilate_iterations: u32,
}

impl Default for DetectPreprocessParams {
    fn default() -> Self {
        Self {
            blur_size: 5,
            adaptive_block_size: 15,
            adaptive_c: 4.0,
            dilate_kernel_w: 20,
            dilate_kernel_h: 3,
            dilate_iterations: 2,
        }
    }
}

impl DetectPreprocessKernel {
    /// Load the detection preprocessing PTX for the detected GPU architecture.
    pub fn new(ctx: &GpuContext, arch: GpuArch) -> Result<Self> {
        let ptx = gpu_arch::select_ptx(
            arch,
            include_str!(concat!(env!("OUT_DIR"), "/detect_preprocess_sm80.ptx")),
            include_str!(concat!(env!("OUT_DIR"), "/detect_preprocess_sm120.ptx")),
        );

        if ptx.starts_with("// STUB") {
            return Err(OcrError::Preprocess(
                "detect_preprocess kernel not compiled (CUDA unavailable)".into(),
            ));
        }

        let module = ctx
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|e| OcrError::Cuda(format!("failed to load detect_preprocess PTX: {e}")))?;

        let rgb_to_gray = module
            .load_function("rgb_to_gray")
            .map_err(|e| OcrError::Cuda(format!("missing rgb_to_gray: {e}")))?;
        let gaussian_blur_h = module
            .load_function("gaussian_blur_h")
            .map_err(|e| OcrError::Cuda(format!("missing gaussian_blur_h: {e}")))?;
        let gaussian_blur_v = module
            .load_function("gaussian_blur_v")
            .map_err(|e| OcrError::Cuda(format!("missing gaussian_blur_v: {e}")))?;
        let integral_image_h = module
            .load_function("integral_image_h")
            .map_err(|e| OcrError::Cuda(format!("missing integral_image_h: {e}")))?;
        let integral_image_v = module
            .load_function("integral_image_v")
            .map_err(|e| OcrError::Cuda(format!("missing integral_image_v: {e}")))?;
        let adaptive_threshold_mean = module
            .load_function("adaptive_threshold_mean")
            .map_err(|e| OcrError::Cuda(format!("missing adaptive_threshold_mean: {e}")))?;
        let dilate_h = module
            .load_function("dilate_h")
            .map_err(|e| OcrError::Cuda(format!("missing dilate_h: {e}")))?;
        let dilate_v = module
            .load_function("dilate_v")
            .map_err(|e| OcrError::Cuda(format!("missing dilate_v: {e}")))?;

        tracing::debug!(arch = arch.name(), "loaded detect_preprocess PTX kernel");

        Ok(Self {
            rgb_to_gray,
            gaussian_blur_h,
            gaussian_blur_v,
            integral_image_h,
            integral_image_v,
            adaptive_threshold_mean,
            dilate_h,
            dilate_v,
            _module: module,
        })
    }

    /// Process RGB u8 HWC image on GPU, returning binary masks on CPU.
    ///
    /// All image processing runs on GPU (grayscale, blur, threshold, dilation).
    /// Returns `(dilated_binary, pre_dilation_binary)`:
    /// - `dilated_binary`: for CCL (characters connected into lines)
    /// - `pre_dilation_binary`: for edge fragment trimming
    pub fn execute(
        &self,
        ctx: &GpuContext,
        rgb: &[u8],
        w: u32,
        h: u32,
        params: &DetectPreprocessParams,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let width = w as i32;
        let height = h as i32;
        let n = (w * h) as usize;
        let stream = &ctx.default_stream;

        let cfg_2d = LaunchConfig {
            grid_dim: (w.div_ceil(32), h.div_ceil(32), 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };

        // Upload RGB data to GPU.
        let rgb_gpu = stream
            .clone_htod(rgb)
            .map_err(|e| OcrError::Cuda(format!("RGB upload failed: {e}")))?;

        // Allocate grayscale buffer.
        let mut gray = stream
            .alloc_zeros::<u8>(n)
            .map_err(|e| OcrError::Cuda(format!("gray alloc failed: {e}")))?;

        // Step 1: RGB → grayscale.
        unsafe {
            stream
                .launch_builder(&self.rgb_to_gray)
                .arg(&rgb_gpu)
                .arg(&mut gray)
                .arg(&width)
                .arg(&height)
                .launch(cfg_2d)
                .map_err(|e| OcrError::Cuda(format!("rgb_to_gray launch failed: {e}")))?;
        }
        drop(rgb_gpu);

        // Step 2: Gaussian blur (separable, 2 passes).
        if params.blur_size >= 3 {
            let mut blur_tmp = stream
                .alloc_zeros::<u8>(n)
                .map_err(|e| OcrError::Cuda(format!("blur_tmp alloc failed: {e}")))?;

            // Horizontal pass: gray → blur_tmp
            unsafe {
                stream
                    .launch_builder(&self.gaussian_blur_h)
                    .arg(&gray)
                    .arg(&mut blur_tmp)
                    .arg(&width)
                    .arg(&height)
                    .launch(cfg_2d)
                    .map_err(|e| OcrError::Cuda(format!("gaussian_blur_h launch failed: {e}")))?;
            }

            // Vertical pass: blur_tmp → gray (reuse gray as output)
            unsafe {
                stream
                    .launch_builder(&self.gaussian_blur_v)
                    .arg(&blur_tmp)
                    .arg(&mut gray)
                    .arg(&width)
                    .arg(&height)
                    .launch(cfg_2d)
                    .map_err(|e| OcrError::Cuda(format!("gaussian_blur_v launch failed: {e}")))?;
            }
            drop(blur_tmp);
        }

        // Step 3: Integral image (2 passes: row-wise then column-wise).
        let mut integral = stream
            .alloc_zeros::<u32>(n)
            .map_err(|e| OcrError::Cuda(format!("integral alloc failed: {e}")))?;

        let cfg_rows = LaunchConfig {
            grid_dim: (h.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.integral_image_h)
                .arg(&gray)
                .arg(&mut integral)
                .arg(&width)
                .arg(&height)
                .launch(cfg_rows)
                .map_err(|e| OcrError::Cuda(format!("integral_image_h launch failed: {e}")))?;
        }

        let cfg_cols = LaunchConfig {
            grid_dim: (w.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.integral_image_v)
                .arg(&mut integral)
                .arg(&width)
                .arg(&height)
                .launch(cfg_cols)
                .map_err(|e| OcrError::Cuda(format!("integral_image_v launch failed: {e}")))?;
        }

        // Step 4: Adaptive threshold.
        let mut binary = stream
            .alloc_zeros::<u8>(n)
            .map_err(|e| OcrError::Cuda(format!("binary alloc failed: {e}")))?;

        let block_size = params.adaptive_block_size as i32;
        let c_param = params.adaptive_c;
        unsafe {
            stream
                .launch_builder(&self.adaptive_threshold_mean)
                .arg(&gray)
                .arg(&integral)
                .arg(&mut binary)
                .arg(&width)
                .arg(&height)
                .arg(&block_size)
                .arg(&c_param)
                .launch(cfg_2d)
                .map_err(|e| {
                    OcrError::Cuda(format!("adaptive_threshold_mean launch failed: {e}"))
                })?;
        }
        drop(gray);
        drop(integral);

        // Download pre-dilation binary for edge fragment trimming.
        let pre_dilation: Vec<u8> = stream
            .clone_dtoh(&binary)
            .map_err(|e| OcrError::Cuda(format!("pre-dilation download failed: {e}")))?;

        // Step 5: Separable dilation (multiple iterations).
        let kernel_w = params.dilate_kernel_w as i32;
        let kernel_h = params.dilate_kernel_h as i32;

        let mut current = binary;
        for _ in 0..params.dilate_iterations {
            let mut h_dilated = stream
                .alloc_zeros::<u8>(n)
                .map_err(|e| OcrError::Cuda(format!("h_dilated alloc failed: {e}")))?;

            // Horizontal dilation: current → h_dilated
            unsafe {
                stream
                    .launch_builder(&self.dilate_h)
                    .arg(&current)
                    .arg(&mut h_dilated)
                    .arg(&width)
                    .arg(&height)
                    .arg(&kernel_w)
                    .launch(cfg_2d)
                    .map_err(|e| OcrError::Cuda(format!("dilate_h launch failed: {e}")))?;
            }
            drop(current);

            let mut v_dilated = stream
                .alloc_zeros::<u8>(n)
                .map_err(|e| OcrError::Cuda(format!("v_dilated alloc failed: {e}")))?;

            // Vertical dilation: h_dilated → v_dilated
            unsafe {
                stream
                    .launch_builder(&self.dilate_v)
                    .arg(&h_dilated)
                    .arg(&mut v_dilated)
                    .arg(&width)
                    .arg(&height)
                    .arg(&kernel_h)
                    .launch(cfg_2d)
                    .map_err(|e| OcrError::Cuda(format!("dilate_v launch failed: {e}")))?;
            }
            drop(h_dilated);

            current = v_dilated;
        }

        // Step 6: Download dilated binary mask to CPU (implicit stream sync).
        let dilated: Vec<u8> = stream
            .clone_dtoh(&current)
            .map_err(|e| OcrError::Cuda(format!("binary mask download failed: {e}")))?;

        Ok((dilated, pre_dilation))
    }
}
