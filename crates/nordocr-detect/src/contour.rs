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
