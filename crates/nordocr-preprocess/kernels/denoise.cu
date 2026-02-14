// Denoising kernels optimized for scanned document images.
// Includes non-local means and bilateral filter variants.

extern "C" {

// Fast bilateral filter for edge-preserving denoising.
// Preserves text edges while smoothing scan noise.
__global__ void bilateral_filter(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int radius,
    float sigma_spatial,
    float sigma_range
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float center_val = (float)input[y * width + x];
    float weight_sum = 0.0f;
    float filtered = 0.0f;

    float inv_2_sigma_s2 = -0.5f / (sigma_spatial * sigma_spatial);
    float inv_2_sigma_r2 = -0.5f / (sigma_range * sigma_range);

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                continue;

            float neighbor = (float)input[ny * width + nx];

            // Spatial weight.
            float dist2 = (float)(dx * dx + dy * dy);
            float ws = expf(dist2 * inv_2_sigma_s2);

            // Range (intensity) weight.
            float diff = neighbor - center_val;
            float wr = expf(diff * diff * inv_2_sigma_r2);

            float w = ws * wr;
            filtered += neighbor * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = (unsigned char)(filtered / weight_sum);
}

// Median filter (3x3) for salt-and-pepper noise common in fax/scan artifacts.
// Uses a sorting network for the 9 elements.
__global__ void median_filter_3x3(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Collect 3x3 neighborhood.
    unsigned char vals[9];
    int idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            vals[idx++] = input[ny * width + nx];
        }
    }

    // Sorting network for 9 elements (find median).
    #define SWAP(a, b) { if (vals[a] > vals[b]) { unsigned char t = vals[a]; vals[a] = vals[b]; vals[b] = t; } }
    SWAP(0, 1); SWAP(3, 4); SWAP(6, 7);
    SWAP(1, 2); SWAP(4, 5); SWAP(7, 8);
    SWAP(0, 1); SWAP(3, 4); SWAP(6, 7);
    SWAP(0, 3); SWAP(3, 6); SWAP(0, 3);
    SWAP(1, 4); SWAP(4, 7); SWAP(1, 4);
    SWAP(2, 5); SWAP(5, 8); SWAP(2, 5);
    SWAP(1, 3); SWAP(5, 7);
    SWAP(2, 6); SWAP(4, 6); SWAP(2, 4);
    SWAP(2, 3); SWAP(5, 6);
    #undef SWAP

    output[y * width + x] = vals[4]; // median
}

// Gaussian blur for general pre-smoothing.
// 5x5 kernel with sigma ≈ 1.0.
__global__ void gaussian_blur_5x5(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Pre-computed 5x5 Gaussian kernel (sigma ≈ 1.0), sum = 273.
    const int kernel[5][5] = {
        {1,  4,  7,  4,  1},
        {4, 16, 26, 16,  4},
        {7, 26, 41, 26,  7},
        {4, 16, 26, 16,  4},
        {1,  4,  7,  4,  1}
    };

    int sum = 0;
    for (int ky = 0; ky < 5; ky++) {
        for (int kx = 0; kx < 5; kx++) {
            int nx = min(max(x + kx - 2, 0), width - 1);
            int ny = min(max(y + ky - 2, 0), height - 1);
            sum += (int)input[ny * width + nx] * kernel[ky][kx];
        }
    }

    output[y * width + x] = (unsigned char)(sum / 273);
}

} // extern "C"
