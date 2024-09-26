#include <cuda.h>

template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;

    const uint tx = threadIdx.x % BN;
    const uint ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    const uint AxIdx = threadIdx.x % BK;
    const uint AyIdx = threadIdx.x / BK;
    const uint BxIdx = threadIdx.x % BN;
    const uint ByIdx = threadIdx.x / BN;

    float tmps[TM] = { 0.0f };

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[AyIdx * BK + AxIdx] = A[AyIdx * K + AxIdx];
        Bs[ByIdx * BN + BxIdx] = B[ByIdx * N + BxIdx];
        
        __syncthreads();

        for (uint i = 0; i < BK; ++ i) {
            float tmpBs = Bs[i * BN + tx];
            for (uint localTy = 0; localTy < TM; ++ localTy) {
                tmps[localTy] += As[(ty + localTy) * BK + i] * tmpBs;
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for (int localTy = 0; localTy < TM; ++ localTy) {
        C[(ty + localTy) * N + tx] = alpha * tmps[localTy] + beta * C[(ty + localTy) * N + tx];
    }
}