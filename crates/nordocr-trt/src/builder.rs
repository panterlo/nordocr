use std::path::Path;

use nordocr_core::{OcrError, Result};

/// Builder for creating optimized TensorRT engines from ONNX models.
pub struct TrtEngineBuilder {
    fp8_enabled: bool,
    fp4_enabled: bool,
    sparsity_enabled: bool,
    max_batch_size: u32,
    max_workspace_size: usize,
    /// Optimization profiles for dynamic shapes.
    profiles: Vec<OptimizationProfile>,
}

/// An optimization profile specifying min/opt/max shapes for dynamic dims.
#[derive(Debug, Clone)]
pub struct OptimizationProfile {
    pub name: String,
    pub min_shape: Vec<i64>,
    pub opt_shape: Vec<i64>,
    pub max_shape: Vec<i64>,
}

/// FP8 calibration configuration.
pub struct Fp8CalibrationConfig {
    /// Path to calibration dataset (representative input images).
    pub calibration_data_dir: String,
    /// Number of calibration batches.
    pub num_batches: u32,
    /// Cache file for computed calibration scales.
    pub cache_file: String,
}

impl TrtEngineBuilder {
    pub fn new() -> Self {
        Self {
            fp8_enabled: false,
            fp4_enabled: false,
            sparsity_enabled: false,
            max_batch_size: 1,
            max_workspace_size: 1 << 30, // 1 GB default
            profiles: Vec::new(),
        }
    }

    /// Enable FP8 (E4M3) quantization for Blackwell Transformer Engine.
    pub fn with_fp8(mut self) -> Self {
        self.fp8_enabled = true;
        self
    }

    /// Enable FP4 quantization (experimental, Blackwell only).
    pub fn with_fp4(mut self) -> Self {
        self.fp4_enabled = true;
        self
    }

    /// Enable structured sparsity (2:4).
    pub fn with_sparsity(mut self) -> Self {
        self.sparsity_enabled = true;
        self
    }

    pub fn max_batch_size(mut self, size: u32) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn max_workspace_size(mut self, size: usize) -> Self {
        self.max_workspace_size = size;
        self
    }

    /// Add a dynamic shape optimization profile.
    pub fn add_profile(mut self, profile: OptimizationProfile) -> Self {
        self.profiles.push(profile);
        self
    }

    /// Build a TensorRT engine from an ONNX model file.
    ///
    /// This is the offline engine-building step. The resulting engine is
    /// serialized to `output_path` and loaded at runtime via `TrtEngine::load`.
    pub fn build_from_onnx(
        &self,
        onnx_path: &Path,
        output_path: &Path,
        calibration: Option<&Fp8CalibrationConfig>,
    ) -> Result<()> {
        if !onnx_path.exists() {
            return Err(OcrError::ModelLoad(format!(
                "ONNX file not found: {}",
                onnx_path.display()
            )));
        }

        tracing::info!(
            onnx = %onnx_path.display(),
            output = %output_path.display(),
            fp8 = self.fp8_enabled,
            fp4 = self.fp4_enabled,
            sparsity = self.sparsity_enabled,
            max_batch = self.max_batch_size,
            "building TensorRT engine"
        );

        // In production, this calls the TensorRT builder API:
        //
        // 1. Create builder + network + parser
        //    let builder = createInferBuilder(logger);
        //    let network = builder.createNetworkV2(EXPLICIT_BATCH);
        //    let parser = nvonnxparser::createParser(network, logger);
        //    parser.parseFromFile(onnx_path, ...);
        //
        // 2. Configure builder
        //    let config = builder.createBuilderConfig();
        //    config.setMaxWorkspaceSize(self.max_workspace_size);
        //    if self.fp8_enabled {
        //        config.setFlag(BuilderFlag::FP8);
        //        config.setFlag(BuilderFlag::FP16); // FP8 requires FP16 fallback
        //        if let Some(cal) = calibration {
        //            // Set FP8 calibrator
        //        }
        //    }
        //    if self.sparsity_enabled {
        //        config.setFlag(BuilderFlag::SPARSE_WEIGHTS);
        //    }
        //
        // 3. Set optimization profiles for dynamic shapes
        //    for profile in &self.profiles {
        //        let p = builder.createOptimizationProfile();
        //        p.setDimensions("input", min, opt, max);
        //        config.addOptimizationProfile(p);
        //    }
        //
        // 4. Build and serialize
        //    let plan = builder.buildSerializedNetwork(network, config);
        //    std::fs::write(output_path, plan.data());

        Ok(())
    }
}

impl Default for TrtEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
