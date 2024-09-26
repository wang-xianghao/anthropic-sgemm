#include <cuda.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float *A,
                                     const float *B, float beta, float *C) {
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    const uint tx = (threadIdx.x % (BN / TN)) * TN;
    const uint ty = (threadIdx.x / (BN / TN)) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    const uint AxIdx = threadIdx.x % BK;
    const uint AyIdx = threadIdx.x / BK;
    const uint strideA = blockDim.x / BK;

    const uint BxIdx = threadIdx.x % BN;
    const uint ByIdx = threadIdx.x / BN;
    const uint strideB = blockDim.x / BN;

    float tmps[TM * TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset < BM; offset += strideA) {
            As[(AyIdx + offset) * BK + AxIdx] = A[(AyIdx + offset) * K + AxIdx];
        }

        for (uint offset = 0; offset < BK; offset += strideB) {
            Bs[(ByIdx + offset) * BN + BxIdx] = B[(ByIdx + offset) * N + BxIdx];
        }

        __syncthreads();

        for (uint i = 0; i < BK; ++i) {
            for (uint j = 0; j < TM; ++ j) {
                regA[j] = As[(ty + j) * BK + i];
            }

            for (uint j = 0; j < TN; ++ j) {
                regB[TN] = Bs[i * BN + tx + j];
            }

            for (uint localTy = 0; localTy < TM; ++localTy) {
                for (uint localTx = 0; localTx < TN; ++localTx) {
                    tmps[localTy * TN + localTx] += regA[i] * regB[i];
                }
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for (uint localTy = 0; localTy < TM; ++localTy) {
        for (uint localTx = 0; localTx < TN; ++localTx) {
            C[(ty + localTy) * N + tx + localTx] = alpha * tmps[localTy * TN + localTx] + beta * C[(ty + localTy) * N + tx + localTx];
        }
    }
}