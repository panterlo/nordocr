//! Integration test: morphological text detection on a real document page.
//!
//! Uses a PNG extracted from a multi-page TIFF (Fax4 G4 compression is not
//! supported by the `tiff` 0.9 crate, so we use a pre-extracted PNG).

use std::path::Path;

#[test]
fn test_morphological_on_document_page() {
    let png_path = Path::new("/home/sysop/nordocr/test_page5.png");
    if !png_path.exists() {
        eprintln!("Skipping: test image not found at {}", png_path.display());
        return;
    }

    eprintln!("Loading {}", png_path.display());
    let img = image::open(png_path).expect("failed to load image");
    let rgb = img.to_rgb8();
    let width = rgb.width();
    let height = rgb.height();
    let rgb_data = rgb.into_raw();
    let page_index = 4u32;

    eprintln!("  Image: {}x{}, {} bytes RGB", width, height, rgb_data.len());

    // Run morphological detection with default parameters.
    let detector = nordocr_detect::MorphologicalDetector::default();
    let start = std::time::Instant::now();
    let regions = detector.detect(&rgb_data, width, height, page_index);
    let elapsed = start.elapsed();

    eprintln!("\nMorphological detection results:");
    eprintln!(
        "  Detected {} text regions in {:.1}ms",
        regions.len(),
        elapsed.as_secs_f64() * 1000.0
    );
    eprintln!(
        "  Params: kernel={}x{}, iter={}, thresh={}, min={}x{}, pad={}",
        detector.kernel_w,
        detector.kernel_h,
        detector.iterations,
        detector.threshold,
        detector.min_width,
        detector.min_height,
        detector.region_padding,
    );

    eprintln!(
        "\n  {:>3}  {:>7} {:>7} {:>7} {:>6}",
        "#", "x", "y", "w", "h"
    );
    eprintln!("  {}", "-".repeat(40));
    for (i, r) in regions.iter().enumerate() {
        eprintln!(
            "  {:3}  {:7.1} {:7.1} {:7.1} {:6.1}",
            i + 1,
            r.bbox.x,
            r.bbox.y,
            r.bbox.width,
            r.bbox.height,
        );
    }

    // Sanity checks for a financial document page.
    assert!(
        !regions.is_empty(),
        "Should detect text regions on a financial document"
    );
    assert!(
        regions.len() > 5,
        "Financial page should have many text lines, got {}",
        regions.len()
    );

    // Verify regions are sorted top-to-bottom, left-to-right.
    for pair in regions.windows(2) {
        let a = &pair[0].bbox;
        let b = &pair[1].bbox;
        assert!(
            a.y < b.y || (a.y == b.y && a.x <= b.x),
            "Regions should be sorted: ({:.0},{:.0}) before ({:.0},{:.0})",
            a.x,
            a.y,
            b.x,
            b.y,
        );
    }

    // Check no region is absurdly large (shouldn't span the whole page).
    for r in &regions {
        assert!(
            r.bbox.width < width as f32 * 0.95,
            "Region too wide: {:.0}px (page is {}px)",
            r.bbox.width,
            width,
        );
    }

    eprintln!("\nTest passed: {} regions detected", regions.len());
}
