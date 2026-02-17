//! Integration test: GPU vs CPU morphological detection comparison.
//!
//! Loads all pages from a pre-extracted PNG directory (from a multi-page TIFF),
//! runs both CPU and GPU detection paths, compares results, and benchmarks.

use std::path::Path;
use std::time::Instant;

use nordocr_detect::{GpuMorphologicalDetector, MorphologicalDetector};
use nordocr_gpu::{GpuContext, GpuContextConfig};

/// Load all page_XX.png files from a directory, sorted by name.
fn load_pages(dir: &Path) -> Vec<(Vec<u8>, u32, u32)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("failed to read page directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "png" || ext == "PNG")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    entries
        .iter()
        .map(|entry| {
            let img = image::open(entry.path()).expect("failed to load PNG");
            let rgb = img.to_rgb8();
            (rgb.clone().into_raw(), rgb.width(), rgb.height())
        })
        .collect()
}

#[test]
fn test_gpu_vs_cpu_detection_all_pages() {
    let pages_dir = Path::new(r"c:\temp\tiff_pages");
    if !pages_dir.exists() {
        eprintln!("Skipping: page directory not found at {}", pages_dir.display());
        return;
    }

    let pages = load_pages(pages_dir);
    eprintln!("Loaded {} pages", pages.len());
    assert!(pages.len() >= 18, "expected 18 pages, got {}", pages.len());

    // Initialize GPU context.
    let gpu = GpuContext::new(GpuContextConfig {
        device_ordinal: 0,
        pool_size: 128 * 1024 * 1024,
        stream_count: 1,
    })
    .expect("failed to init GPU");

    // Initialize detectors.
    let cpu_detector = MorphologicalDetector::default();
    let gpu_detector =
        GpuMorphologicalDetector::new(&gpu).expect("failed to init GPU detector");

    let mut total_cpu_ms = 0.0f64;
    let mut total_gpu_ms = 0.0f64;
    let mut total_cpu_regions = 0usize;
    let mut total_gpu_regions = 0usize;

    eprintln!(
        "\n{:>5}  {:>6}  {:>10}  {:>6}  {:>10}  {:>7}  {:>5}",
        "Page", "CPU#", "CPU ms", "GPU#", "GPU ms", "Speedup", "Match"
    );
    eprintln!("{}", "-".repeat(65));

    for (page_idx, (rgb, width, height)) in pages.iter().enumerate() {
        // CPU detection.
        let cpu_start = Instant::now();
        let cpu_regions = cpu_detector.detect(rgb, *width, *height, page_idx as u32);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // GPU detection.
        let gpu_start = Instant::now();
        let gpu_regions = gpu_detector
            .detect(&gpu, rgb, *width, *height, page_idx as u32)
            .expect("GPU detection failed");
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms;
        let count_match = cpu_regions.len() == gpu_regions.len();

        eprintln!(
            "{:5}  {:6}  {:10.1}  {:6}  {:10.1}  {:7.1}x  {:>5}",
            page_idx,
            cpu_regions.len(),
            cpu_ms,
            gpu_regions.len(),
            gpu_ms,
            speedup,
            if count_match { "OK" } else { "DIFF" }
        );

        total_cpu_ms += cpu_ms;
        total_gpu_ms += gpu_ms;
        total_cpu_regions += cpu_regions.len();
        total_gpu_regions += gpu_regions.len();

        // Verify approximate match.
        let count_diff = (cpu_regions.len() as i32 - gpu_regions.len() as i32).abs();
        assert!(
            count_diff <= 3,
            "Page {}: region count mismatch too large: CPU={}, GPU={} (diff={})",
            page_idx,
            cpu_regions.len(),
            gpu_regions.len(),
            count_diff
        );
    }

    eprintln!("{}", "-".repeat(65));
    eprintln!(
        "{:>5}  {:6}  {:10.1}  {:6}  {:10.1}  {:7.1}x",
        "TOTAL",
        total_cpu_regions,
        total_cpu_ms,
        total_gpu_regions,
        total_gpu_ms,
        total_cpu_ms / total_gpu_ms
    );
    eprintln!(
        "\nSummary: {} pages, CPU {:.0}ms, GPU {:.0}ms, {:.1}x speedup",
        pages.len(),
        total_cpu_ms,
        total_gpu_ms,
        total_cpu_ms / total_gpu_ms
    );
    eprintln!(
        "Regions: CPU={}, GPU={} (C# reference: 876)",
        total_cpu_regions, total_gpu_regions
    );
}
