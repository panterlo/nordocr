// GPU-accelerated preprocessing kernels for morphological text detection.
//
// Pipeline: RGB → grayscale → Gaussian blur → integral image → adaptive threshold → dilation
// All kernels operate on u8 images and run sequentially on the same CUDA stream.

extern "C" {

// Convert RGB HWC u8 image to grayscale u8 using BT.601 luma coefficients.
// Matches the CPU path: (r*77 + g*150 + b*29) >> 8
__global__ void rgb_to_gray(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ gray,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int rgb_idx = idx * 3;
    unsigned int r = rgb[rgb_idx];
    unsigned int g = rgb[rgb_idx + 1];
    unsigned int b = rgb[rgb_idx + 2];
    gray[idx] = (unsigned char)((r * 77 + g * 150 + b * 29) >> 8);
}

// Horizontal pass of separable Gaussian blur with 5-tap binomial kernel [1,4,6,4,1]/16.
// Approximates Gaussian with sigma ~1.1 (OpenCV ksize=5 default).
// Uses clamped (replicate) boundary conditions.
__global__ void gaussian_blur_h(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int row = y * width;

    int x0 = max(x - 2, 0);
    int x1 = max(x - 1, 0);
    int x3 = min(x + 1, width - 1);
    int x4 = min(x + 2, width - 1);

    unsigned int acc = (unsigned int)input[row + x0] * 1
                     + (unsigned int)input[row + x1] * 4
                     + (unsigned int)input[row + x]  * 6
                     + (unsigned int)input[row + x3] * 4
                     + (unsigned int)input[row + x4] * 1;

    output[row + x] = (unsigned char)(acc / 16);
}

// Vertical pass of separable Gaussian blur with 5-tap binomial kernel [1,4,6,4,1]/16.
__global__ void gaussian_blur_v(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y0 = max(y - 2, 0);
    int y1 = max(y - 1, 0);
    int y3 = min(y + 1, height - 1);
    int y4 = min(y + 2, height - 1);

    unsigned int acc = (unsigned int)input[y0 * width + x] * 1
                     + (unsigned int)input[y1 * width + x] * 4
                     + (unsigned int)input[y  * width + x] * 6
                     + (unsigned int)input[y3 * width + x] * 4
                     + (unsigned int)input[y4 * width + x] * 1;

    output[y * width + x] = (unsigned char)(acc / 16);
}

// Row-wise prefix sum for integral image computation.
// One thread per row — sequential within row, parallel across rows.
// Output type is u32 to handle sums up to 255 * width without overflow
// (safe for width <= 16M pixels, well beyond any document page).
__global__ void integral_image_h(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ integral,
    int width,
    int height
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    unsigned int sum = 0;
    for (int x = 0; x < width; x++) {
        sum += input[y * width + x];
        integral[y * width + x] = sum;
    }
}

// Column-wise prefix sum to complete 2D integral image.
// One thread per column — sequential within column, parallel across columns.
__global__ void integral_image_v(
    unsigned int* __restrict__ integral,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    for (int y = 1; y < height; y++) {
        integral[y * width + x] += integral[(y - 1) * width + x];
    }
}

// Adaptive mean thresholding using integral (summed-area) image.
// Matches OpenCV ADAPTIVE_THRESH_MEAN_C with THRESH_BINARY_INV:
//   threshold = local_mean - C
//   output = (pixel <= threshold) ? 255 : 0
// Dark text on light background → white foreground.
__global__ void adaptive_threshold_mean(
    const unsigned char* __restrict__ gray,
    const unsigned int* __restrict__ integral,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int block_size,
    float c_param
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = block_size / 2;

    // Window bounds (inclusive), clamped to image.
    // We use SAT convention: sum of rect [x0..x1, y0..y1] inclusive
    // = integral[y1][x1] - integral[y0-1][x1] - integral[y1][x0-1] + integral[y0-1][x0-1]
    int x0 = max(x - half, 0);
    int y0 = max(y - half, 0);
    int x1 = min(x + half, width - 1);
    int y1 = min(y + half, height - 1);

    float area = (float)((x1 - x0 + 1) * (y1 - y0 + 1));

    // SAT lookup with boundary handling (no -1 border needed).
    float sum = (float)integral[y1 * width + x1];
    if (x0 > 0) sum -= (float)integral[y1 * width + (x0 - 1)];
    if (y0 > 0) sum -= (float)integral[(y0 - 1) * width + x1];
    if (x0 > 0 && y0 > 0) sum += (float)integral[(y0 - 1) * width + (x0 - 1)];

    float mean = sum / area;
    float thresh = mean - c_param;

    // BINARY_INV: dark text (below threshold) → white foreground.
    output[y * width + x] = ((float)gray[y * width + x] <= thresh) ? 255 : 0;
}

// Horizontal binary dilation (sliding window maximum along rows).
// For each pixel, output is 255 if any pixel in [x - half_w, x + half_w] is > 0.
__global__ void dilate_h(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int kernel_w
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_w = kernel_w / 2;
    int row = y * width;

    int x0 = max(x - half_w, 0);
    int x1 = min(x + half_w, width - 1);

    unsigned char max_val = 0;
    for (int kx = x0; kx <= x1; kx++) {
        unsigned char val = input[row + kx];
        if (val > max_val) max_val = val;
    }

    output[row + x] = max_val;
}

// Vertical binary dilation (sliding window maximum along columns).
__global__ void dilate_v(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int kernel_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_h = kernel_h / 2;

    int y0 = max(y - half_h, 0);
    int y1 = min(y + half_h, height - 1);

    unsigned char max_val = 0;
    for (int ky = y0; ky <= y1; ky++) {
        unsigned char val = input[ky * width + x];
        if (val > max_val) max_val = val;
    }

    output[y * width + x] = max_val;
}

} // extern "C"
