#include <cuda.h>

template <const int BLOCK_SIZE>
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint BM = BLOCK_SIZE;
    const uint BN = BLOCK_SIZE;
    const uint BK = BLOCK_SIZE;

    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    const uint tx = threadIdx.x % BN;
    const uint ty = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];

        __syncthreads();

        for (int i = 0; i < BK; ++ i) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads();

        A += BK;
        B += BK * N;
    }

    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}