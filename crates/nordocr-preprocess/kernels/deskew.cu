// Deskew kernel for correcting rotation in scanned documents.
// Uses projection profile analysis + affine warp.

#include <math.h>

extern "C" {

// Compute horizontal projection profile (row pixel sums).
// Used to find the skew angle via variance maximization.
__global__ void projection_profile(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ profile,
    int width,
    int height
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    unsigned int sum = 0;
    for (int x = 0; x < width; x++) {
        sum += (input[y * width + x] == 0) ? 1 : 0;  // count black pixels
    }
    profile[y] = sum;
}

// Compute variance of projection profile for a given rotation angle.
// Higher variance → more distinct text lines → better alignment.
// Called for each candidate angle in the search range.
__global__ void profile_variance(
    const unsigned int* __restrict__ profiles,
    float* __restrict__ variances,
    int height,
    int num_angles
) {
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (angle_idx >= num_angles) return;

    const unsigned int* profile = profiles + angle_idx * height;

    // Compute mean.
    float sum = 0.0f;
    for (int y = 0; y < height; y++) {
        sum += (float)profile[y];
    }
    float mean = sum / (float)height;

    // Compute variance.
    float var_sum = 0.0f;
    for (int y = 0; y < height; y++) {
        float diff = (float)profile[y] - mean;
        var_sum += diff * diff;
    }
    variances[angle_idx] = var_sum / (float)height;
}

// Affine warp (rotation) using bilinear interpolation.
// Rotates the image by `angle` radians around its center.
__global__ void affine_rotate(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    float cos_angle,
    float sin_angle,
    float cx,
    float cy
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Inverse rotation to find source coordinates.
    float dx = (float)x - cx;
    float dy = (float)y - cy;
    float src_x = cos_angle * dx + sin_angle * dy + cx;
    float src_y = -sin_angle * dx + cos_angle * dy + cy;

    // Bilinear interpolation.
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = src_x - (float)x0;
    float fy = src_y - (float)y0;

    // Boundary check — fill with white (255) outside image.
    auto sample = [&](int sx, int sy) -> float {
        if (sx < 0 || sx >= width || sy < 0 || sy >= height)
            return 255.0f;
        return (float)input[sy * width + sx];
    };

    float val = sample(x0, y0) * (1.0f - fx) * (1.0f - fy)
              + sample(x1, y0) * fx * (1.0f - fy)
              + sample(x0, y1) * (1.0f - fx) * fy
              + sample(x1, y1) * fx * fy;

    output[y * width + x] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
}

} // extern "C"
