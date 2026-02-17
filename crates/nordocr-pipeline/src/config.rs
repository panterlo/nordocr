use serde::{Deserialize, Serialize};

/// Detection model architecture.
///
/// Morphological is the default for scanned documents — CPU-based, no model needed.
/// DBNet++ and RTMDet are neural alternatives requiring TensorRT engines.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DetectModelArch {
    /// CPU morphological detection: binarize → dilate → CCL → bbox filtering.
    /// Ported from Ormeo.Document C# pipeline. No GPU model needed.
    /// Best for clean scanned documents with horizontal text.
    #[default]
    Morphological,
    /// DBNet++ with differentiable binarization. Proven on document benchmarks.
    /// Single-pass, fully convolutional, no NMS needed. Requires TRT engine.
    DBNetPP,
    /// RTMDet (Real-Time Models for Object Detection). MMLAB's latest
    /// single-stage detector. Potentially faster than DBNet++ with CSPNeXt
    /// backbone, but less tested on document text detection specifically.
    /// Export: PyTorch → ONNX → TensorRT, same as DBNet++. Requires TRT engine.
    RTMDet,
}

/// Recognition model architecture.
///
/// SVTRv2 is the default: CTC decoder with variable-width input, best for
/// wide text lines in scanned documents. PARSeq is the alternative with
/// fixed-width input and parallel autoregressive decoding.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecogModelArch {
    /// SVTRv2 with CTC decoder. Variable-width input (stride 4), handles
    /// arbitrarily wide text lines without padding waste. Trained at 1792px
    /// max width for Nordic documents. CER 0.20%, accuracy 98.45%.
    #[default]
    SVTRv2,
    /// PARSeq (Permutation AutoRegressive Sequence model). Non-autoregressive:
    /// all character positions decoded in parallel. Fixed 384px input width.
    /// CER 0.14%, accuracy 99.07%. Better on short/medium lines.
    PARSeq,
    /// MAERec (Masked Autoencoder for scene text Recognition). Uses MAE
    /// pre-training for stronger visual features. May outperform PARSeq on
    /// degraded/noisy scans. Autoregressive decoding — slower inference
    /// but potentially higher accuracy on hard cases.
    MAERec,
    /// CLIP4STR (CLIP for Scene Text Recognition). Leverages CLIP's
    /// vision-language pre-training for robust character recognition.
    /// Strongest zero-shot generalization to unseen fonts/styles.
    /// Heavier model (~100M+ params vs PARSeq's ~20M) — use only if
    /// accuracy on unusual fonts justifies the throughput cost.
    CLIP4STR,
}

/// TensorRT precision mode for inference engines.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum InferencePrecision {
    /// FP32 — maximum accuracy, lowest throughput. Debugging only.
    FP32,
    /// FP16 — good accuracy/speed tradeoff. Works on all GPUs.
    /// Use this on A6000 Ampere (sm_86).
    #[default]
    FP16,
    /// FP8 (E4M3) — ~2x throughput over FP16. Requires Blackwell (sm_120)
    /// or Hopper (sm_90). Needs calibration data for quantization.
    FP8,
    /// FP4 (NF4) — experimental, ~2x throughput over FP8 on weight-bound
    /// layers. Blackwell (sm_120) only. May degrade accuracy on small
    /// models (PARSeq-S has only ~20M params — limited redundancy to
    /// absorb quantization error). Best suited for detection backbone
    /// where the model is larger.
    FP4,
}

/// DLA (Deep Learning Accelerator) offload configuration.
///
/// Blackwell GPUs include dedicated DLA cores that can run inference
/// independently of the GPU's SM cores. By offloading detection or
/// recognition to DLA, the GPU SMs remain free for preprocessing
/// CUDA kernels — enabling true pipeline parallelism.
///
/// Not available on A6000 Ada. On Blackwell, there are typically 2 DLA cores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlaOffloadConfig {
    /// Offload detection model to DLA.
    pub offload_detect: bool,
    /// Offload recognition model to DLA.
    pub offload_recognize: bool,
    /// DLA core index (0 or 1).
    pub dla_core: u32,
    /// Allow GPU fallback for layers the DLA doesn't support.
    pub allow_gpu_fallback: bool,
}

impl Default for DlaOffloadConfig {
    fn default() -> Self {
        Self {
            offload_detect: false,
            offload_recognize: false,
            dla_core: 0,
            allow_gpu_fallback: true,
        }
    }
}

/// Runtime configuration for the OCR pipeline.
///
/// # GPU targets
///
/// TensorRT engines are GPU-architecture-specific and must be rebuilt
/// for each target GPU. The engine paths should point to files built
/// for the GPU this process will run on:
///
/// - **Development (A6000 Ampere, sm_86)**: `models/detect_sm86.engine`
/// - **Production (RTX 6000 PRO Blackwell, sm_120)**: `models/detect_sm120.engine`
///
/// CUDA kernels (preprocessing) are compiled as fat binaries containing
/// code for both architectures — no configuration needed there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Path to the detection TensorRT engine file.
    /// Only needed when detect_model is DBNetPP or RTMDet.
    /// Must be built for the target GPU architecture.
    pub detect_engine_path: Option<String>,
    /// Path to the recognition TensorRT engine file.
    /// Must be built for the target GPU architecture.
    pub recognize_engine_path: String,

    // Model architecture selection.
    /// Detection model architecture.
    pub detect_model: DetectModelArch,
    /// Recognition model architecture.
    pub recognize_model: RecogModelArch,
    /// Inference precision for TensorRT engines.
    pub precision: InferencePrecision,

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
    /// Whether to use NVIDIA DALI for GPU-accelerated image decode.
    /// Falls back to nvJPEG + image crate if DALI is unavailable.
    pub enable_dali: bool,
    /// DLA offload configuration (Blackwell only).
    pub dla: DlaOffloadConfig,
    /// Override GPU architecture instead of auto-detecting.
    /// Values: "sm_80" (Ampere/Ada) or "sm_120" (RTX 6000 PRO Blackwell).
    /// None = auto-detect from CUDA device properties.
    pub gpu_arch_override: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detect_engine_path: None, // Not needed for Morphological detection
            recognize_engine_path: "models/recognize_svtrv2_sm120.engine".to_string(),

            detect_model: DetectModelArch::default(),
            recognize_model: RecogModelArch::default(),
            precision: InferencePrecision::default(),

            detect_max_batch: 4,
            detect_input_height: 1024,
            detect_input_width: 1024,
            detect_threshold: 0.3,
            detect_min_area: 100.0,

            recognize_max_batch: 64,
            recognize_input_height: 32,
            recognize_max_input_width: 1792,
            recognize_max_seq_len: 448, // 1792 / 4 (SVTRv2 stride)

            num_streams: 4,
            gpu_pool_size: 256 * 1024 * 1024, // 256 MB
            enable_cuda_graph: true,
            enable_preprocess: true,
            enable_dali: false,
            dla: DlaOffloadConfig::default(),
            gpu_arch_override: None,
        }
    }
}

/// Configuration presets for known GPU targets.
impl PipelineConfig {
    /// Preset for development on A6000 Ampere (sm_86, 48 GB VRAM).
    ///
    /// Uses FP16 precision (FP8/FP4 not available on Ampere).
    /// No DLA (Ampere doesn't have dedicated DLA cores).
    /// Smaller batch sizes to match Ampere's SM count.
    pub fn a6000_ada() -> Self {
        Self {
            detect_engine_path: None,
            recognize_engine_path: "models/recognize_svtrv2_768_sm86.engine".to_string(),
            gpu_arch_override: Some("sm_86".to_string()),
            precision: InferencePrecision::FP16,
            detect_max_batch: 2,
            recognize_max_batch: 32,
            num_streams: 2,
            dla: DlaOffloadConfig::default(), // DLA not available on Ada
            ..Self::default()
        }
    }

    /// Preset for production on RTX 6000 PRO Blackwell (sm_120, 96 GB VRAM).
    ///
    /// Uses FP8 precision for ~2x throughput over FP16.
    /// Larger batch sizes to saturate Blackwell's SM count.
    /// DLA offload available but disabled by default (enable after benchmarking).
    pub fn rtx6000_pro_blackwell() -> Self {
        Self {
            detect_engine_path: None,
            recognize_engine_path: "models/recognize_svtrv2_sm120.engine".to_string(),
            gpu_arch_override: Some("sm_120".to_string()),
            precision: InferencePrecision::FP8,
            detect_max_batch: 8,
            recognize_max_batch: 128,
            num_streams: 4,
            gpu_pool_size: 512 * 1024 * 1024, // 512 MB
            ..Self::default()
        }
    }

    /// Experimental: Blackwell with DLA offload.
    ///
    /// Runs detection on DLA core 0, freeing all GPU SMs for preprocessing
    /// and recognition. Theoretical benefit: preprocessing CUDA kernels run
    /// concurrently with detection inference on separate hardware.
    pub fn rtx6000_pro_blackwell_dla() -> Self {
        Self {
            dla: DlaOffloadConfig {
                offload_detect: true,
                offload_recognize: false,
                dla_core: 0,
                allow_gpu_fallback: true,
            },
            ..Self::rtx6000_pro_blackwell()
        }
    }

    /// Experimental: Maximum throughput on Blackwell with FP4 + DLA.
    ///
    /// Uses FP4 (NF4) quantization for recognition + DLA for detection.
    /// WARNING: FP4 on PARSeq-S (~20M params) may degrade diacritical
    /// accuracy. Validate on the Nordic test set before deploying.
    pub fn rtx6000_pro_blackwell_experimental() -> Self {
        Self {
            precision: InferencePrecision::FP4,
            dla: DlaOffloadConfig {
                offload_detect: true,
                offload_recognize: false,
                dla_core: 0,
                allow_gpu_fallback: true,
            },
            enable_dali: true,
            recognize_max_batch: 256,
            ..Self::rtx6000_pro_blackwell()
        }
    }
}
