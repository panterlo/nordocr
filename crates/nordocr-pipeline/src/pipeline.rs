use std::path::Path;
use std::time::Instant;

use nordocr_core::{FileInput, OcrError, PageResult, RawImage, Result, TextRegion, TimingInfo};
use nordocr_decode::Decoder;
use nordocr_detect::{DetectionBatcher, DetectionEngine, DetectionPostprocessor, GpuMorphologicalDetector, MorphologicalDetector};
use nordocr_gpu::{GpuContext, GpuContextConfig};
use nordocr_preprocess::{PreprocessConfig, PreprocessPipeline};
use nordocr_recognize::{RecognitionBatcher, RecognitionEngine};

use crate::config::{DetectModelArch, PipelineConfig};
use crate::scheduler::PageScheduler;

/// The full OCR pipeline: decode → detect → recognize.
///
/// Detection backend: morphological (CPU or GPU) or neural (TRT).
enum DetectionBackend {
    Morphological(MorphologicalDetector),
    GpuMorphological(GpuMorphologicalDetector),
    Neural(DetectionBatcher),
}

pub struct OcrPipeline {
    gpu: GpuContext,
    decoder: Decoder,
    preprocess: Option<PreprocessPipeline>,
    detection: DetectionBackend,
    recognizer: RecognitionBatcher,
    scheduler: PageScheduler,
    config: PipelineConfig,
}

impl OcrPipeline {
    /// Build the pipeline from configuration.
    pub fn build(config: PipelineConfig) -> Result<Self> {
        tracing::info!("building OCR pipeline");

        let gpu = GpuContext::new(GpuContextConfig {
            device_ordinal: 0,
            pool_size: config.gpu_pool_size,
            stream_count: config.num_streams,
        })?;

        let decoder = Decoder::new(&gpu)?;

        let preprocess = if config.enable_preprocess {
            Some(PreprocessPipeline::new(
                &gpu,
                PreprocessConfig::default(),
            )?)
        } else {
            None
        };

        let detection = match config.detect_model {
            DetectModelArch::Morphological => {
                // Try GPU-accelerated morphological detection first.
                match GpuMorphologicalDetector::new(&gpu) {
                    Ok(gpu_det) => {
                        tracing::info!("using morphological text detection (GPU)");
                        DetectionBackend::GpuMorphological(gpu_det)
                    }
                    Err(e) => {
                        tracing::info!(
                            reason = %e,
                            "GPU morphological detection unavailable, falling back to CPU"
                        );
                        DetectionBackend::Morphological(MorphologicalDetector::default())
                    }
                }
            }
            DetectModelArch::DBNetPP | DetectModelArch::RTMDet => {
                let engine_path = config.detect_engine_path.as_deref().ok_or_else(|| {
                    OcrError::InvalidInput(
                        "detect_engine_path required for neural detection".into(),
                    )
                })?;
                let detect_engine = DetectionEngine::load(
                    &gpu,
                    Path::new(engine_path),
                    config.detect_max_batch,
                    config.detect_input_height,
                    config.detect_input_width,
                )?;
                let detect_postproc = DetectionPostprocessor::new()
                    .with_threshold(config.detect_threshold)
                    .with_min_area(config.detect_min_area);
                DetectionBackend::Neural(DetectionBatcher::new(detect_engine, detect_postproc))
            }
        };

        let recog_engine = RecognitionEngine::load(
            &gpu,
            Path::new(&config.recognize_engine_path),
            config.recognize_max_batch,
            config.recognize_input_height,
            config.recognize_max_input_width,
            config.recognize_max_seq_len,
        )?;
        let recognizer = RecognitionBatcher::new(recog_engine, config.recognize_max_batch);

        let scheduler = PageScheduler::new(config.num_streams);

        tracing::info!("OCR pipeline ready");

        Ok(Self {
            gpu,
            decoder,
            preprocess,
            detection,
            recognizer,
            scheduler,
            config,
        })
    }

    /// Process a file input through the full OCR pipeline.
    pub fn process(
        &mut self,
        input: &FileInput,
        page_filter: Option<&[u32]>,
    ) -> Result<(Vec<PageResult>, TimingInfo)> {
        let total_start = Instant::now();
        let mut timing = TimingInfo::default();

        // Stage 1: Decode input to CPU RGB images.
        // For morphological detection, we need CPU data; for neural, we'd use GPU.
        // Using CPU decode for both paths since recognition also needs CPU crops.
        let decode_start = Instant::now();
        let page_images = self.decoder.decode_cpu(input, page_filter)?;
        timing.decode_ms = decode_start.elapsed().as_secs_f32() * 1000.0;

        if page_images.is_empty() {
            return Ok((Vec::new(), timing));
        }

        tracing::debug!(num_pages = page_images.len(), "decoded input");

        // Stage 2: Text detection.
        let detect_start = Instant::now();
        let regions_per_page = self.detect_all(&page_images)?;
        timing.detect_ms = detect_start.elapsed().as_secs_f32() * 1000.0;

        let total_regions: usize = regions_per_page.iter().map(|r| r.len()).sum();
        tracing::debug!(
            total_regions,
            pages = page_images.len(),
            "detection complete"
        );

        // Stage 3: Collect all text regions and run recognition.
        let recog_start = Instant::now();
        let all_regions: Vec<TextRegion> = regions_per_page
            .iter()
            .flat_map(|r| r.iter().cloned())
            .collect();

        let text_lines = if !all_regions.is_empty() {
            self.recognizer
                .recognize_all(&self.gpu, &all_regions, &page_images)?
        } else {
            Vec::new()
        };
        timing.recognize_ms = recog_start.elapsed().as_secs_f32() * 1000.0;

        // Stage 4: Assemble page results.
        let mut page_results = Vec::with_capacity(page_images.len());
        let mut line_cursor = 0;

        for (page_idx, regions) in regions_per_page.iter().enumerate() {
            let page_lines: Vec<_> = text_lines
                .get(line_cursor..line_cursor + regions.len())
                .unwrap_or_default()
                .to_vec();

            let full_text = page_lines
                .iter()
                .map(|l| l.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            let avg_confidence = if page_lines.is_empty() {
                0.0
            } else {
                page_lines.iter().map(|l| l.confidence).sum::<f32>() / page_lines.len() as f32
            };

            page_results.push(PageResult {
                page_index: page_idx as u32,
                text: full_text,
                lines: page_lines,
                confidence: avg_confidence,
            });

            line_cursor += regions.len();
        }

        timing.total_ms = total_start.elapsed().as_secs_f32() * 1000.0;

        tracing::info!(
            pages = page_results.len(),
            total_ms = timing.total_ms,
            decode_ms = timing.decode_ms,
            detect_ms = timing.detect_ms,
            recog_ms = timing.recognize_ms,
            "pipeline complete"
        );

        Ok((page_results, timing))
    }

    /// Run text detection on all pages.
    fn detect_all(&mut self, page_images: &[RawImage]) -> Result<Vec<Vec<TextRegion>>> {
        match &self.detection {
            DetectionBackend::Morphological(detector) => {
                let mut all_regions = Vec::with_capacity(page_images.len());
                for (page_idx, page) in page_images.iter().enumerate() {
                    let regions =
                        detector.detect(&page.data, page.width, page.height, page_idx as u32);
                    tracing::debug!(
                        page = page_idx,
                        regions = regions.len(),
                        "morphological detection (CPU)"
                    );
                    all_regions.push(regions);
                }
                Ok(all_regions)
            }
            DetectionBackend::GpuMorphological(detector) => {
                let mut all_regions = Vec::with_capacity(page_images.len());
                for (page_idx, page) in page_images.iter().enumerate() {
                    let regions = detector.detect(
                        &self.gpu,
                        &page.data,
                        page.width,
                        page.height,
                        page_idx as u32,
                    )?;
                    tracing::debug!(
                        page = page_idx,
                        regions = regions.len(),
                        "morphological detection (GPU)"
                    );
                    all_regions.push(regions);
                }
                Ok(all_regions)
            }
            DetectionBackend::Neural(_batcher) => {
                // Neural detection requires GPU-preprocessed images.
                // For now, return an error since we don't have GPU tensors here.
                Err(OcrError::InvalidInput(
                    "neural detection requires GPU preprocessing (not yet wired with CPU decode path)".into(),
                ))
            }
        }
    }

    /// Warm up the pipeline by running a dummy inference.
    pub fn warmup(&mut self) -> Result<()> {
        tracing::info!("warming up pipeline");

        let dummy = vec![128u8; 100 * 100];
        let input = FileInput::Image(dummy);

        let _ = self.process(&input, None);

        self.gpu.synchronize()?;
        tracing::info!("warmup complete");
        Ok(())
    }
}
