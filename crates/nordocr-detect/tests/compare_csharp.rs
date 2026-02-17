//! Compare Rust GPU morphological detection against C# .NET pipeline results.
//!
//! Loads the C# `ocr_results.json`, runs GPU detection on extracted PNG pages,
//! then matches regions by proximity (center distance) and reports deviations.

use std::path::Path;

use nordocr_detect::{GpuMorphologicalDetector, MorphologicalDetector};
use nordocr_gpu::{GpuContext, GpuContextConfig};
use serde::Deserialize;

/// C# pipeline result entry.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CSharpRegion {
    page_number: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    text: String,
    text_confidence: f64,
    column: i32,
    row: i32,
    density: f64,
    cluster_id: i32,
}

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
fn compare_gpu_vs_csharp() {
    let pages_dir = Path::new(r"c:\temp\tiff_pages");
    let csharp_json = Path::new(r"C:\Temp\tesseract-pipeline\ocr_results.json");

    if !pages_dir.exists() || !csharp_json.exists() {
        eprintln!(
            "Skipping: need {} and {}",
            pages_dir.display(),
            csharp_json.display()
        );
        return;
    }

    // Load C# reference results.
    let json_str =
        std::fs::read_to_string(csharp_json).expect("failed to read ocr_results.json");
    let csharp_regions: Vec<CSharpRegion> =
        serde_json::from_str(&json_str).expect("failed to parse JSON");
    eprintln!("C# reference: {} total regions", csharp_regions.len());

    // Group C# regions by page.
    let max_page = csharp_regions
        .iter()
        .map(|r| r.page_number)
        .max()
        .unwrap_or(0);
    let mut csharp_by_page: Vec<Vec<&CSharpRegion>> = vec![vec![]; (max_page + 1) as usize];
    for r in &csharp_regions {
        csharp_by_page[r.page_number as usize].push(r);
    }

    // Load pages.
    let pages = load_pages(pages_dir);
    eprintln!("Loaded {} pages\n", pages.len());

    // Initialize GPU.
    let gpu = GpuContext::new(GpuContextConfig {
        device_ordinal: 0,
        pool_size: 128 * 1024 * 1024,
        stream_count: 1,
    })
    .expect("failed to init GPU");

    let gpu_detector =
        GpuMorphologicalDetector::new(&gpu).expect("failed to init GPU detector");

    // Per-page comparison.
    let mut total_rust = 0usize;
    let mut total_csharp = 0usize;
    let mut total_matched = 0usize;
    let mut total_large_deviations = 0usize;
    let mut all_unmatched_csharp: Vec<(u32, &CSharpRegion)> = Vec::new();
    let mut all_unmatched_rust: Vec<(u32, nordocr_core::BBox)> = Vec::new();

    eprintln!(
        "{:>5}  {:>6}  {:>6}  {:>7}  {:>7}  {:>9}",
        "Page", "C#", "Rust", "Match", "Unm-C#", "Unm-Rust"
    );
    eprintln!("{}", "-".repeat(55));

    for (page_idx, (rgb, width, height)) in pages.iter().enumerate() {
        let rust_regions = gpu_detector
            .detect(&gpu, rgb, *width, *height, page_idx as u32)
            .expect("GPU detection failed");

        let cs_page = if page_idx < csharp_by_page.len() {
            &csharp_by_page[page_idx]
        } else {
            &vec![]
        };

        // Match regions by center proximity (greedy nearest-neighbor).
        // For each C# region, find the closest Rust region within 50px center distance.
        let mut rust_matched = vec![false; rust_regions.len()];
        let mut cs_matched = vec![false; cs_page.len()];
        let mut page_matched = 0usize;

        // Build Rust centers.
        let rust_centers: Vec<(f32, f32)> = rust_regions
            .iter()
            .map(|r| (r.bbox.x + r.bbox.width / 2.0, r.bbox.y + r.bbox.height / 2.0))
            .collect();

        // Greedy match: sort by distance, match closest pairs first.
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (ci, cs) in cs_page.iter().enumerate() {
            let cs_cx = cs.x as f32 + cs.width as f32 / 2.0;
            let cs_cy = cs.y as f32 + cs.height as f32 / 2.0;
            for (ri, &(rx, ry)) in rust_centers.iter().enumerate() {
                let dist = ((cs_cx - rx).powi(2) + (cs_cy - ry).powi(2)).sqrt();
                if dist < 100.0 {
                    pairs.push((ci, ri, dist));
                }
            }
        }
        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Track matched pairs with deviations.
        struct MatchInfo {
            cs_idx: usize,
            rust_idx: usize,
            dx: f32,
            dy: f32,
            dw: f32,
            dh: f32,
            dist: f32,
        }
        let mut matches: Vec<MatchInfo> = Vec::new();

        for (ci, ri, dist) in &pairs {
            if cs_matched[*ci] || rust_matched[*ri] {
                continue;
            }
            cs_matched[*ci] = true;
            rust_matched[*ri] = true;
            page_matched += 1;

            let cs = &cs_page[*ci];
            let rr = &rust_regions[*ri];

            matches.push(MatchInfo {
                cs_idx: *ci,
                rust_idx: *ri,
                dx: rr.bbox.x - cs.x as f32,
                dy: rr.bbox.y - cs.y as f32,
                dw: rr.bbox.width - cs.width as f32,
                dh: rr.bbox.height - cs.height as f32,
                dist: *dist,
            });
        }

        let unmatched_cs: Vec<usize> = (0..cs_page.len())
            .filter(|i| !cs_matched[*i])
            .collect();
        let unmatched_rust: Vec<usize> = (0..rust_regions.len())
            .filter(|i| !rust_matched[*i])
            .collect();

        eprintln!(
            "{:5}  {:6}  {:6}  {:7}  {:7}  {:9}",
            page_idx,
            cs_page.len(),
            rust_regions.len(),
            page_matched,
            unmatched_cs.len(),
            unmatched_rust.len(),
        );

        // Report large deviations for this page (>20px in any dimension).
        let mut page_large = 0;
        for m in &matches {
            if m.dx.abs() > 20.0 || m.dy.abs() > 20.0 || m.dw.abs() > 40.0 || m.dh.abs() > 40.0
            {
                page_large += 1;
                if page_large <= 5 {
                    let cs = &cs_page[m.cs_idx];
                    let rr = &rust_regions[m.rust_idx];
                    eprintln!(
                        "   DEVIATION p{}: C#({},{},{}x{}) vs Rust({:.0},{:.0},{:.0}x{:.0})  Δx={:+.0} Δy={:+.0} Δw={:+.0} Δh={:+.0}  text=\"{}\"",
                        page_idx,
                        cs.x, cs.y, cs.width, cs.height,
                        rr.bbox.x, rr.bbox.y, rr.bbox.width, rr.bbox.height,
                        m.dx, m.dy, m.dw, m.dh,
                        &cs.text[..cs.text.len().min(40)]
                    );
                }
            }
        }
        if page_large > 5 {
            eprintln!("   ... and {} more large deviations on page {}", page_large - 5, page_idx);
        }
        total_large_deviations += page_large;

        // Track unmatched regions.
        for &ci in &unmatched_cs {
            all_unmatched_csharp.push((page_idx as u32, cs_page[ci]));
        }
        for &ri in &unmatched_rust {
            all_unmatched_rust.push((page_idx as u32, rust_regions[ri].bbox));
        }

        total_rust += rust_regions.len();
        total_csharp += cs_page.len();
        total_matched += page_matched;
    }

    eprintln!("{}", "-".repeat(55));
    eprintln!(
        "TOTAL  {:6}  {:6}  {:7}  {:7}  {:9}",
        total_csharp,
        total_rust,
        total_matched,
        all_unmatched_csharp.len(),
        all_unmatched_rust.len(),
    );

    // Summary stats.
    eprintln!("\n=== SUMMARY ===");
    eprintln!(
        "C# regions: {}, Rust GPU regions: {}, Matched: {} ({:.1}%)",
        total_csharp,
        total_rust,
        total_matched,
        total_matched as f64 / total_csharp as f64 * 100.0
    );
    eprintln!(
        "Large position deviations (>20px pos or >40px size): {}",
        total_large_deviations
    );

    // List unmatched C# regions (regions C# found but Rust missed).
    if !all_unmatched_csharp.is_empty() {
        eprintln!("\n--- Unmatched C# regions (Rust missed) ---");
        for (page, cs) in &all_unmatched_csharp {
            eprintln!(
                "  p{}: ({},{},{}x{}) row={} col={} text=\"{}\"",
                page,
                cs.x,
                cs.y,
                cs.width,
                cs.height,
                cs.row,
                cs.column,
                &cs.text[..cs.text.len().min(50)]
            );
        }
    }

    // List unmatched Rust regions (Rust found but C# didn't).
    if !all_unmatched_rust.is_empty() {
        eprintln!("\n--- Unmatched Rust regions (C# missed) ---");
        for (page, bbox) in &all_unmatched_rust {
            eprintln!(
                "  p{}: ({:.0},{:.0},{:.0}x{:.0})",
                page, bbox.x, bbox.y, bbox.width, bbox.height
            );
        }
    }
}
