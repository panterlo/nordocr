use nordocr_core::BBox;

/// Info about a connected component extracted from a binary image.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    pub bbox: BBox,
    /// Principal axis orientation in degrees [0, 180).
    /// 0° = horizontal, 90° = vertical.
    pub orientation_deg: f32,
    /// Number of foreground pixels in the component.
    pub pixel_count: u32,
}

/// Extract bounding boxes of connected components from a binary u8 image.
///
/// White pixels (> 0) are foreground, black (0) is background.
/// Uses 4-connected two-pass CCL with union-find (path halving).
pub fn extract_bboxes(binary: &[u8], width: u32, height: u32) -> Vec<BBox> {
    extract_components(binary, width, height)
        .into_iter()
        .map(|c| c.bbox)
        .collect()
}

/// Extract connected components with bounding boxes, orientation, and pixel count.
///
/// White pixels (> 0) are foreground, black (0) is background.
/// Uses 4-connected two-pass CCL with union-find (path halving).
/// Orientation is computed from second-order central moments of each component.
pub fn extract_components(binary: &[u8], width: u32, height: u32) -> Vec<ComponentInfo> {
    let w = width as usize;
    let h = height as usize;

    if w == 0 || h == 0 {
        return Vec::new();
    }

    let mut labels = vec![0u32; w * h];
    // parent[0] = 0 (background, unused). Grows as labels are assigned.
    let mut parent: Vec<u32> = vec![0];
    let mut next_label: u32 = 1;

    // --- Pass 1: assign provisional labels ---
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if binary[idx] == 0 {
                continue;
            }

            let left = if x > 0 { labels[idx - 1] } else { 0 };
            let top = if y > 0 { labels[idx - w] } else { 0 };

            match (left > 0, top > 0) {
                (false, false) => {
                    labels[idx] = next_label;
                    parent.push(next_label); // self-referencing root
                    next_label += 1;
                }
                (true, false) => {
                    labels[idx] = left;
                }
                (false, true) => {
                    labels[idx] = top;
                }
                (true, true) => {
                    let rl = find(&mut parent, left);
                    let rt = find(&mut parent, top);
                    let min_l = rl.min(rt);
                    labels[idx] = min_l;
                    if rl != rt {
                        let max_l = rl.max(rt);
                        parent[max_l as usize] = min_l;
                    }
                }
            }
        }
    }

    // Flatten all parent pointers to roots.
    for i in 1..parent.len() {
        parent[i] = find(&mut parent, i as u32);
    }

    // --- Pass 2: compute bounding boxes and moments ---
    // Per-component accumulators: (min_x, min_y, max_x, max_y, sum_x, sum_y, sum_xx, sum_yy, sum_xy, count)
    struct Accum {
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
        sum_x: f64,
        sum_y: f64,
        sum_xx: f64,
        sum_yy: f64,
        sum_xy: f64,
        count: u32,
    }

    let mut accum_map: Vec<Option<Accum>> = (0..parent.len()).map(|_| None).collect();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = labels[idx];
            if label == 0 {
                continue;
            }
            let root = parent[label as usize] as usize;
            let xu = x as u32;
            let yu = y as u32;
            let xf = x as f64;
            let yf = y as f64;
            match accum_map[root] {
                Some(ref mut a) => {
                    a.min_x = a.min_x.min(xu);
                    a.min_y = a.min_y.min(yu);
                    a.max_x = a.max_x.max(xu);
                    a.max_y = a.max_y.max(yu);
                    a.sum_x += xf;
                    a.sum_y += yf;
                    a.sum_xx += xf * xf;
                    a.sum_yy += yf * yf;
                    a.sum_xy += xf * yf;
                    a.count += 1;
                }
                None => {
                    accum_map[root] = Some(Accum {
                        min_x: xu,
                        min_y: yu,
                        max_x: xu,
                        max_y: yu,
                        sum_x: xf,
                        sum_y: yf,
                        sum_xx: xf * xf,
                        sum_yy: yf * yf,
                        sum_xy: xf * yf,
                        count: 1,
                    });
                }
            }
        }
    }

    accum_map
        .into_iter()
        .flatten()
        .map(|a| {
            let bbox = BBox::new(
                a.min_x as f32,
                a.min_y as f32,
                (a.max_x - a.min_x + 1) as f32,
                (a.max_y - a.min_y + 1) as f32,
            );

            // Compute orientation from central moments.
            let n = a.count as f64;
            let cx = a.sum_x / n;
            let cy = a.sum_y / n;
            let mu20 = a.sum_xx / n - cx * cx;
            let mu02 = a.sum_yy / n - cy * cy;
            let mu11 = a.sum_xy / n - cx * cy;

            // Principal axis angle: 0.5 * atan2(2*mu11, mu20 - mu02)
            // Returns angle in [-90, 90) degrees from horizontal.
            let angle_rad = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);
            // Normalize to [0, 180).
            let mut angle_deg = angle_rad.to_degrees();
            if angle_deg < 0.0 {
                angle_deg += 180.0;
            }

            ComponentInfo {
                bbox,
                orientation_deg: angle_deg as f32,
                pixel_count: a.count,
            }
        })
        .collect()
}

/// Split components that are significantly taller than the typical line height.
///
/// When dilation bridges adjacent text lines, the resulting component spans
/// multiple lines. This function detects such merges by comparing component
/// height against the median and splits them at horizontal gaps in the
/// binary mask's foreground projection profile.
///
/// `max_height_ratio`: components taller than `median_height * ratio` are split.
/// Typical value: 1.8 (catches 2-line merges and beyond).
pub fn split_tall_components(
    components: Vec<ComponentInfo>,
    binary: &[u8],
    width: u32,
    height: u32,
    max_height_ratio: f32,
) -> Vec<ComponentInfo> {
    if components.len() < 3 {
        return components;
    }

    // Compute median height of reasonably-sized components.
    let mut heights: Vec<f32> = components
        .iter()
        .filter(|c| c.bbox.height >= 10.0)
        .map(|c| c.bbox.height)
        .collect();

    if heights.len() < 3 {
        return components;
    }

    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_h = heights[heights.len() / 2];
    let split_threshold = median_h * max_height_ratio;

    let w = width as usize;
    let h = height as usize;
    let mut result = Vec::with_capacity(components.len());

    for comp in components {
        if comp.bbox.height <= split_threshold {
            result.push(comp);
            continue;
        }

        // This component is too tall — try to split it.
        let x0 = (comp.bbox.x as usize).min(w.saturating_sub(1));
        let y0 = (comp.bbox.y as usize).min(h.saturating_sub(1));
        let bw = (comp.bbox.width as usize).min(w - x0);
        let bh = (comp.bbox.height as usize).min(h - y0);

        if bh < 10 || bw < 5 {
            result.push(comp);
            continue;
        }

        // Compute horizontal projection (count foreground pixels per row).
        let mut projection = vec![0u32; bh];
        for dy in 0..bh {
            let row_start = (y0 + dy) * w + x0;
            let mut count = 0u32;
            for dx in 0..bw {
                if binary[row_start + dx] > 0 {
                    count += 1;
                }
            }
            projection[dy] = count;
        }

        let max_proj = *projection.iter().max().unwrap_or(&1);
        if max_proj == 0 {
            result.push(comp);
            continue;
        }

        // Find valleys: rows where projection < 30% of max.
        let valley_thresh = (max_proj as f32 * 0.3) as u32;

        // Don't split within the top or bottom 20% of the component.
        let margin = bh / 5;
        let search_start = margin.max(1);
        let search_end = bh.saturating_sub(margin).max(search_start + 1);

        // Find valley segments (consecutive rows below threshold).
        let mut valleys: Vec<(usize, usize)> = Vec::new();
        let mut in_valley = false;
        let mut valley_start = 0;

        for dy in search_start..search_end {
            if projection[dy] <= valley_thresh {
                if !in_valley {
                    valley_start = dy;
                    in_valley = true;
                }
            } else if in_valley {
                valleys.push((valley_start, dy));
                in_valley = false;
            }
        }
        if in_valley {
            valleys.push((valley_start, search_end));
        }

        if valleys.is_empty() {
            // No clear valley — keep as is.
            result.push(comp);
            continue;
        }

        // Use valley midpoints as split coordinates.
        let mut split_ys: Vec<usize> = valleys.iter().map(|(s, e)| (s + e) / 2).collect();
        split_ys.sort();

        // Create sub-components.
        let min_sub_h = 5usize;
        let mut prev_dy = 0usize;
        for &split_dy in &split_ys {
            let sub_h = split_dy - prev_dy;
            if sub_h >= min_sub_h {
                result.push(ComponentInfo {
                    bbox: BBox::new(
                        comp.bbox.x,
                        (y0 + prev_dy) as f32,
                        comp.bbox.width,
                        sub_h as f32,
                    ),
                    orientation_deg: comp.orientation_deg,
                    pixel_count: projection[prev_dy..split_dy].iter().sum(),
                });
            }
            prev_dy = split_dy;
        }
        // Final segment.
        let sub_h = bh - prev_dy;
        if sub_h >= min_sub_h {
            result.push(ComponentInfo {
                bbox: BBox::new(
                    comp.bbox.x,
                    (y0 + prev_dy) as f32,
                    comp.bbox.width,
                    sub_h as f32,
                ),
                orientation_deg: comp.orientation_deg,
                pixel_count: projection[prev_dy..bh].iter().sum(),
            });
        }
    }

    result
}

/// Trim small edge fragments from components using the pre-dilation binary.
///
/// When horizontal dilation bridges separate text elements (e.g., a note number "N"
/// merged with "Balansräkning"), the CCL bbox includes both. This function uses the
/// vertical projection profile of the *pre-dilation* binary to find content runs,
/// clusters nearby runs (gap < 8px = same cluster), and trims edge clusters that
/// are tiny compared to the main body.
///
/// A leading/trailing cluster is trimmed when:
/// - It is separated from the next cluster by a gap ≥ 8px (wider than inter-char gaps)
/// - The cluster is narrower than 80px
/// - The cluster is less than 1/4 the width of the remaining content
/// Trim small edge fragments from detected regions using pre-dilation binary.
///
/// Strategy: cluster nearby content columns into word-level groups (gap < 4px),
/// then find the widest gap between word-clusters. If the widest gap is near an
/// edge and the edge content is a small fragment, trim it.
///
/// This handles garbage characters merged into bboxes by horizontal dilation —
/// e.g. a "D" from a margin label merged into "Förvaltningsberättelse".
pub fn trim_edge_fragments(
    components: Vec<ComponentInfo>,
    pre_dilation_binary: &[u8],
    width: u32,
    height: u32,
) -> Vec<ComponentInfo> {
    let w = width as usize;
    let h = height as usize;
    // Gap threshold for clustering characters into words.
    // Inter-character gaps are 1-3px; this absorbs them while keeping
    // inter-word gaps (5+px) as separate clusters.
    let char_gap = 4usize;
    let max_fragment_width = 80usize;

    components
        .into_iter()
        .map(|mut comp| {
            let x0 = (comp.bbox.x as usize).min(w.saturating_sub(1));
            let y0 = (comp.bbox.y as usize).min(h.saturating_sub(1));
            let bw = (comp.bbox.width as usize).min(w - x0);
            let bh = (comp.bbox.height as usize).min(h - y0);

            if bw < 60 || bh < 5 {
                return comp;
            }

            // Compute vertical projection (foreground pixels per column) on pre-dilation binary.
            let mut v_proj = vec![0u32; bw];
            for dy in 0..bh {
                let row_start = (y0 + dy) * w + x0;
                for dx in 0..bw {
                    if pre_dilation_binary[row_start + dx] > 0 {
                        v_proj[dx] += 1;
                    }
                }
            }

            // Find content runs (consecutive non-zero columns).
            let mut runs: Vec<(usize, usize)> = Vec::new();
            let mut run_start = None;
            for dx in 0..bw {
                if v_proj[dx] > 0 {
                    if run_start.is_none() {
                        run_start = Some(dx);
                    }
                } else if let Some(start) = run_start {
                    runs.push((start, dx));
                    run_start = None;
                }
            }
            if let Some(start) = run_start {
                runs.push((start, bw));
            }

            if runs.len() < 2 {
                return comp;
            }

            // Cluster runs into word-level groups: merge runs with gap < char_gap.
            let mut words: Vec<(usize, usize)> = Vec::new();
            let mut cur_start = runs[0].0;
            let mut cur_end = runs[0].1;
            for &(rs, re) in &runs[1..] {
                if rs - cur_end < char_gap {
                    cur_end = re;
                } else {
                    words.push((cur_start, cur_end));
                    cur_start = rs;
                    cur_end = re;
                }
            }
            words.push((cur_start, cur_end));

            if words.len() < 2 {
                return comp;
            }

            let mut trim_start = words[0].0;
            let mut trim_end = words.last().unwrap().1;
            let orig_start = trim_start;
            let orig_end = trim_end;

            // --- Check 1: Vertical center-of-mass (catches different-line fragments) ---
            // When dilation bridges text from adjacent lines, the garbage fragment
            // sits at a different vertical position within the merged bbox.
            let vert_threshold = (bh as f32 / 8.0).max(4.0);

            // Check leading fragment.
            if words.len() >= 2 {
                let first_w = words[0].1 - words[0].0;
                let rest_start = words[1].0;
                let rest_end = words.last().unwrap().1;
                let rest_w = rest_end - rest_start;
                if first_w <= max_fragment_width && rest_w >= 32 && first_w * 4 < rest_w {
                    if let (Some(edge_cy), Some(body_cy)) = (
                        vertical_center_of_mass(
                            pre_dilation_binary, w, x0, y0,
                            words[0].0, words[0].1, bh,
                        ),
                        vertical_center_of_mass(
                            pre_dilation_binary, w, x0, y0,
                            rest_start, rest_end, bh,
                        ),
                    ) {
                        if (edge_cy - body_cy).abs() > vert_threshold {
                            trim_start = words[1].0;
                        }
                    }
                }
            }

            // Check trailing fragment.
            if words.len() >= 2 {
                let last = words.last().unwrap();
                let last_w = last.1 - last.0;
                let body_start = trim_start;
                let body_end = words[words.len() - 2].1;
                let body_w = body_end.saturating_sub(body_start);
                if last_w <= max_fragment_width && body_w >= 32 && last_w * 4 < body_w {
                    if let (Some(edge_cy), Some(body_cy)) = (
                        vertical_center_of_mass(
                            pre_dilation_binary, w, x0, y0,
                            last.0, last.1, bh,
                        ),
                        vertical_center_of_mass(
                            pre_dilation_binary, w, x0, y0,
                            body_start, body_end, bh,
                        ),
                    ) {
                        if (edge_cy - body_cy).abs() > vert_threshold {
                            trim_end = words[words.len() - 2].1;
                        }
                    }
                }
            }

            // --- Check 2: Outlier-gap-based trim (catches same-line edge fragments) ---
            // Only for edges not already trimmed by vertical check.
            let mut word_gaps: Vec<(usize, usize)> = Vec::new();
            for i in 0..words.len() - 1 {
                let gap_w = words[i + 1].0 - words[i].1;
                word_gaps.push((i, gap_w));
            }

            let (widest_idx, widest_gap) =
                *word_gaps.iter().max_by_key(|(_, gw)| *gw).unwrap();

            if widest_gap >= 4 {
                let mut is_outlier = true;
                let mut sorted_gaps: Vec<usize> =
                    word_gaps.iter().map(|(_, gw)| *gw).collect();
                sorted_gaps.sort_unstable();

                if sorted_gaps.len() >= 2 {
                    let second_widest = sorted_gaps[sorted_gaps.len() - 2];
                    if second_widest > 0 && widest_gap * 2 < second_widest * 3 {
                        is_outlier = false;
                    }
                }

                if is_outlier {
                    // Check leading fragment (only if not already trimmed).
                    if trim_start == orig_start && widest_idx == 0 {
                        let first_w = words[0].1 - words[0].0;
                        let rest_w = words.last().unwrap().1 - words[1].0;
                        if first_w <= max_fragment_width
                            && rest_w >= 32
                            && first_w * 4 < rest_w
                        {
                            trim_start = words[1].0;
                        }
                    }

                    // Check trailing fragment (only if not already trimmed).
                    if trim_end == orig_end && widest_idx == words.len() - 2 {
                        let last_w =
                            words.last().unwrap().1 - words.last().unwrap().0;
                        let body_w = words[words.len() - 2].1 - trim_start;
                        if last_w <= max_fragment_width
                            && body_w >= 32
                            && last_w * 4 < body_w
                        {
                            trim_end = words[words.len() - 2].1;
                        }
                    }
                }
            }

            // Apply trim if changed.
            let new_w = trim_end.saturating_sub(trim_start);
            if new_w >= 32
                && (trim_start > orig_start || trim_end < orig_end)
            {
                comp.bbox = BBox::new(
                    (x0 + trim_start) as f32,
                    comp.bbox.y,
                    new_w as f32,
                    comp.bbox.height,
                );
            }

            comp
        })
        .collect()
}

/// Compute the vertical center of mass of foreground pixels in a column range.
///
/// Returns the mean y-offset (relative to bbox top) of foreground pixels within
/// columns `[col_start, col_end)` of the bbox starting at `(x0, y0)`.
fn vertical_center_of_mass(
    binary: &[u8],
    full_width: usize,
    x0: usize,
    y0: usize,
    col_start: usize,
    col_end: usize,
    bh: usize,
) -> Option<f32> {
    let mut sum_y = 0.0f64;
    let mut count = 0u32;
    for dy in 0..bh {
        let row_start = (y0 + dy) * full_width + x0;
        for dx in col_start..col_end {
            if binary[row_start + dx] > 0 {
                sum_y += dy as f64;
                count += 1;
            }
        }
    }
    if count > 0 {
        Some((sum_y / count as f64) as f32)
    } else {
        None
    }
}

/// Union-find with path halving.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_image() {
        let img = vec![0u8; 100];
        let bbs = extract_bboxes(&img, 10, 10);
        assert!(bbs.is_empty());
    }

    #[test]
    fn single_pixel() {
        let mut img = vec![0u8; 25];
        img[12] = 255; // center of 5x5
        let bbs = extract_bboxes(&img, 5, 5);
        assert_eq!(bbs.len(), 1);
        assert_eq!(bbs[0].x, 2.0);
        assert_eq!(bbs[0].y, 2.0);
        assert_eq!(bbs[0].width, 1.0);
        assert_eq!(bbs[0].height, 1.0);
    }

    #[test]
    fn two_separate_blobs() {
        // 10x5 image with two blobs
        let mut img = vec![0u8; 50];
        // Blob 1: top-left 2x2
        img[0] = 255;
        img[1] = 255;
        img[10] = 255;
        img[11] = 255;
        // Blob 2: bottom-right 3x1
        img[37] = 255;
        img[38] = 255;
        img[39] = 255;
        let bbs = extract_bboxes(&img, 10, 5);
        assert_eq!(bbs.len(), 2);
    }

    #[test]
    fn split_tall_component() {
        // Simulate two horizontal text lines merged into one tall component.
        // Image: 100x80, two blobs at y=10..30 and y=50..70, gap at y=30..50.
        let w = 100usize;
        let h = 80usize;
        let mut binary = vec![0u8; w * h];

        // Top line: y=10..30
        for y in 10..30 {
            for x in 10..90 {
                binary[y * w + x] = 255;
            }
        }
        // Bottom line: y=50..70
        for y in 50..70 {
            for x in 10..90 {
                binary[y * w + x] = 255;
            }
        }

        // Create one merged component spanning both lines.
        let components = vec![
            ComponentInfo {
                bbox: BBox::new(10.0, 10.0, 80.0, 60.0),
                orientation_deg: 0.0,
                pixel_count: 80 * 40,
            },
            // A normal-height component for median calculation.
            ComponentInfo {
                bbox: BBox::new(10.0, 75.0, 40.0, 20.0),
                orientation_deg: 0.0,
                pixel_count: 40 * 20,
            },
            ComponentInfo {
                bbox: BBox::new(50.0, 75.0, 40.0, 20.0),
                orientation_deg: 0.0,
                pixel_count: 40 * 20,
            },
        ];

        let result = split_tall_components(components, &binary, w as u32, h as u32, 1.8);

        // The tall component (60px) should be split; median is ~20px, threshold ~36px.
        // We should get 2 sub-components from the split + 2 normal ones = 4.
        assert!(
            result.len() >= 4,
            "expected >= 4 components after split, got {}",
            result.len()
        );

        // The two normal components should still be there.
        let normal_count = result.iter().filter(|c| c.bbox.height <= 25.0).count();
        assert!(normal_count >= 2, "normal components should be preserved");

        // The split should produce sub-components shorter than the original 60px.
        let tall_count = result.iter().filter(|c| c.bbox.height > 36.0).count();
        assert_eq!(tall_count, 0, "no component should remain taller than threshold");
    }

    #[test]
    fn l_shaped_blob_merges() {
        // 5x5, L-shape that requires union
        let mut img = vec![0u8; 25];
        // Horizontal bar: row 0, cols 0-3
        for x in 0..4 {
            img[x] = 255;
        }
        // Vertical bar: col 0, rows 0-3
        for y in 0..4 {
            img[y * 5] = 255;
        }
        let bbs = extract_bboxes(&img, 5, 5);
        assert_eq!(bbs.len(), 1);
        assert_eq!(bbs[0].x, 0.0);
        assert_eq!(bbs[0].y, 0.0);
        assert_eq!(bbs[0].width, 4.0);
        assert_eq!(bbs[0].height, 4.0);
    }
}
