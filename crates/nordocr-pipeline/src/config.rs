use serde::{Deserialize, Serialize};

/// Runtime configuration for the OCR pipeline.
///
/// # GPU targets
///
/// TensorRT engines are GPU-architecture-specific and must be rebuilt
/// for each target GPU. The engine paths should point to files built
/// for the GPU this process will run on:
///
/// - **Development (A6000 Ada, sm_89)**: `models/detect_sm89.engine`
/// - **Production (RTX 6000 PRO Blackwell, sm_120)**: `models/detect_sm120.engine`
///
/// CUDA kernels (preprocessing) are compiled as fat binaries containing
/// code for both architectures — no configuration needed there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Path to the detection TensorRT engine file.
    /// Must be built for the target GPU architecture.
    pub detect_engine_path: String,
    /// Path to the recognition TensorRT engine file.
    /// Must be built for the target GPU architecture.
    pub recognize_engine_path: String,

    // Detection settings.
    /// Maximum batch size for detection (pages per batch).
    pub detect_max_batch: u32,
    /// Detection model input height.
    pub detect_input_height: u32,
    /// Detection model input width.
    pub detect_input_width: u32,
    /// Probability threshold for text detection.
    pub detect_threshold: f32,
    /// Minimum area for detected text regions.
    pub detect_min_area: f32,

    // Recognition settings.
    /// Maximum batch size for recognition (text lines per batch).
    pub recognize_max_batch: u32,
    /// Recognition model input height.
    pub recognize_input_height: u32,
    /// Maximum recognition model input width.
    pub recognize_max_input_width: u32,
    /// Maximum output sequence length.
    pub recognize_max_seq_len: u32,

    // Pipeline settings.
    /// Number of CUDA streams for page-level parallelism.
    pub num_streams: usize,
    /// GPU memory pool initial size in bytes.
    pub gpu_pool_size: usize,
    /// Whether to capture the pipeline as a CUDA graph.
    pub enable_cuda_graph: bool,
    /// Whether to run preprocessing.
    pub enable_preprocess: bool,
    /// Override GPU architecture instead of auto-detecting.
    /// Values: "sm_89" (A6000 Ada) or "sm_120" (RTX 6000 PRO Blackwell).
    /// None = auto-detect from CUDA device properties.
    pub gpu_arch_override: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detect_engine_path: "models/detect.engine".to_string(),
            recognize_engine_path: "models/recognize.engine".to_string(),

            detect_max_batch: 4,
            detect_input_height: 1024,
            detect_input_width: 1024,
            detect_threshold: 0.3,
            detect_min_area: 100.0,

            recognize_max_batch: 64,
            recognize_input_height: 32,
            recognize_max_input_width: 512,
            recognize_max_seq_len: 64,

            num_streams: 4,
            gpu_pool_size: 256 * 1024 * 1024, // 256 MB
            enable_cuda_graph: true,
            enable_preprocess: true,
            gpu_arch_override: None,
        }
    }
}

/// Configuration presets for known GPU targets.
impl PipelineConfig {
    /// Preset for development on A6000 Ada (sm_89, 48 GB VRAM).
    pub fn a6000_ada() -> Self {
        Self {
            detect_engine_path: "models/detect_sm89.engine".to_string(),
            recognize_engine_path: "models/recognize_sm89.engine".to_string(),
            gpu_arch_override: Some("sm_89".to_string()),
            // A6000 Ada has fewer SMs than Blackwell — smaller batches.
            detect_max_batch: 2,
            recognize_max_batch: 32,
            num_streams: 2,
            ..Self::default()
        }
    }

    /// Preset for production on RTX 6000 PRO Blackwell (sm_120, 96 GB VRAM).
    pub fn rtx6000_pro_blackwell() -> Self {
        Self {
            detect_engine_path: "models/detect_sm120.engine".to_string(),
            recognize_engine_path: "models/recognize_sm120.engine".to_string(),
            gpu_arch_override: Some("sm_120".to_string()),
            // Blackwell has more SMs, larger L2, higher memory bandwidth.
            detect_max_batch: 8,
            recognize_max_batch: 128,
            num_streams: 4,
            gpu_pool_size: 512 * 1024 * 1024, // 512 MB
            ..Self::default()
        }
    }
}
