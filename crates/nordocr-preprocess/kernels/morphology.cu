// Morphological operations for document image preprocessing.
// Erosion, dilation, opening, closing — used to clean up binarized text.

extern "C" {

// Dilation with a rectangular structuring element.
// Expands bright regions (text thickening in dark-on-light images).
__global__ void dilate_rect(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int kernel_w,
    int kernel_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_kw = kernel_w / 2;
    int half_kh = kernel_h / 2;

    unsigned char max_val = 0;
    for (int ky = -half_kh; ky <= half_kh; ky++) {
        for (int kx = -half_kw; kx <= half_kw; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            unsigned char val = input[ny * width + nx];
            if (val > max_val) max_val = val;
        }
    }

    output[y * width + x] = max_val;
}

// Erosion with a rectangular structuring element.
// Shrinks bright regions (text thinning in dark-on-light images).
__global__ void erode_rect(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int kernel_w,
    int kernel_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_kw = kernel_w / 2;
    int half_kh = kernel_h / 2;

    unsigned char min_val = 255;
    for (int ky = -half_kh; ky <= half_kh; ky++) {
        for (int kx = -half_kw; kx <= half_kw; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            unsigned char val = input[ny * width + nx];
            if (val < min_val) min_val = val;
        }
    }

    output[y * width + x] = min_val;
}

// Connected component labeling (simplified version for small noise removal).
// Labels connected regions of black pixels; small components can be
// filtered out as noise.
// Uses a two-pass approach with equivalence merging.
__global__ void label_init(
    const unsigned char* __restrict__ binary,
    int* __restrict__ labels,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    // Initialize each black pixel with its own label (its linear index).
    // White pixels get label -1.
    labels[idx] = (binary[idx] == 0) ? idx : -1;
}

// Remove small connected components below a pixel count threshold.
// After labeling, count pixels per label and zero out small components.
__global__ void remove_small_components(
    unsigned char* __restrict__ binary,
    const int* __restrict__ labels,
    const int* __restrict__ component_sizes,
    int width,
    int height,
    int min_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int label = labels[idx];

    if (label >= 0 && component_sizes[label] < min_size) {
        binary[idx] = 255;  // remove small component (set to white)
    }
}

} // extern "C"
