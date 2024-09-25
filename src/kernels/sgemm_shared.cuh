#include <cuda.h>
#include <macros.cuh>

template <const int blockSize>
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockSize + threadIdx.x;
    const uint y = blockIdx.y * blockSize + threadIdx.y;

    __shared__ float sA[blockSize][blockSize];
    __shared__ float sB[blockSize][blockSize];

    float tmp = 0.0f;
    for (uint ph = 0; ph < K / blockSize; ++ ph) {
        // Load tile of A
        sA[threadIdx.y][threadIdx.x] = A[y * K + ph * blockSize + threadIdx.x];

        // Load tile of B
        sB[threadIdx.y][threadIdx.x] = B[(ph * blockSize + threadIdx.y) * N + x];

        __syncthreads();

        // Compute the output
        
        for (int i = 0; i < blockSize; ++ i) {
            tmp += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    C[y * N + x] = alpha * tmp + beta * C[y * N + x];
}