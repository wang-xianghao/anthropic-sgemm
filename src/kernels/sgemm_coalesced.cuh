#include <cuda.h>

__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[y * K + i] * B[i * N + x];
        }

        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
    }
}