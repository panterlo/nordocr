// Adaptive binarization kernel optimized for scanned document images.
// Uses Sauvola's method with integral image for O(1) local mean/variance.

extern "C" {

// Compute integral image (prefix sum) for Sauvola binarization.
// This is a two-pass approach: horizontal scan then vertical scan.
__global__ void integral_image_horizontal(
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

__global__ void integral_image_vertical(
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

// Sauvola adaptive binarization.
// For each pixel, computes local mean and standard deviation in a
// window_size x window_size neighborhood using the integral image,
// then applies: threshold = mean * (1 + k * (std/R - 1))
// where k=0.2 and R=128 are typical for scanned documents.
__global__ void adaptive_binarize(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ integral,
    const unsigned long long* __restrict__ integral_sq,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int window_size,
    float k_param,
    float R_param
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_w = window_size / 2;

    // Clamp window to image bounds.
    int x0 = max(0, x - half_w) - 1;
    int y0 = max(0, y - half_w) - 1;
    int x1 = min(width - 1, x + half_w);
    int y1 = min(height - 1, y + half_w);

    // Area of the window.
    float area = (float)((x1 - x0) * (y1 - y0));

    // Sum via integral image: I(x1,y1) - I(x0,y1) - I(x1,y0) + I(x0,y0)
    auto integral_sum = [&](const unsigned int* img, int ax, int ay, int bx, int by) -> unsigned int {
        unsigned int val = img[by * width + bx];
        if (ax >= 0) val -= img[by * width + ax];
        if (ay >= 0) val -= img[ay * width + bx];
        if (ax >= 0 && ay >= 0) val += img[ay * width + ax];
        return val;
    };

    float sum = (float)integral_sum(integral, x0, y0, x1, y1);
    float mean = sum / area;

    // Squared sum for variance (from integral_sq image).
    float sq_sum = (float)integral_sq[y1 * width + x1];
    if (x0 >= 0) sq_sum -= (float)integral_sq[y1 * width + x0];
    if (y0 >= 0) sq_sum -= (float)integral_sq[y0 * width + x1];
    if (x0 >= 0 && y0 >= 0) sq_sum += (float)integral_sq[y0 * width + x0];

    float variance = sq_sum / area - mean * mean;
    float std_dev = sqrtf(fmaxf(variance, 0.0f));

    // Sauvola threshold.
    float threshold = mean * (1.0f + k_param * (std_dev / R_param - 1.0f));

    unsigned char pixel = input[y * width + x];
    output[y * width + x] = (pixel > threshold) ? 255 : 0;
}

} // extern "C"
