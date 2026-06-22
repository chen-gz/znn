#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal. Returns 0 on success, negative error code on failure.
int metal_init(void);

// Perform matrix multiplication on the GPU via Metal
// C = A * B
void metal_matmul(
    int transA, int transB,
    int M, int N, int K,
    const float *A,
    const float *B,
    float *C,
    float beta
);

#ifdef __cplusplus
}
#endif

#endif // METAL_BACKEND_H
