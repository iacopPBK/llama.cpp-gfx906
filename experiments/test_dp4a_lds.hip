// Test: dp4a optimization with LDS (mimics actual flash attention)
// K values loaded from LDS, Q values in registers

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <chrono>

#define CHECK_HIP(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

constexpr int WARP_SIZE = 64;
constexpr int N_WARPS = 4;
constexpr int BLOCK_SIZE = WARP_SIZE * N_WARPS;
constexpr int nbatch_fa = 32;
constexpr int K_row_stride = nbatch_fa + 16;  // +16 padding to avoid bank conflicts
constexpr int DKQ = 64;
constexpr int ncols = 8;

// BASELINE: Current implementation pattern
__global__ void fattn_kq_baseline(
    const int8_t* __restrict__ K_global,
    const half* __restrict__ K_scales_global,
    const int8_t* __restrict__ Q_values,  // [ncols, DKQ]
    const half* __restrict__ Q_scales,    // [2, ncols]
    float* __restrict__ KQ_out            // [nbatch_fa, ncols]
) {
    __shared__ int8_t K_values[nbatch_fa * K_row_stride];
    __shared__ half K_scales[2 * nbatch_fa];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Load K tile to LDS (simplified)
    if (tid < nbatch_fa * 32 / 4) {
        ((int4*)K_values)[tid] = ((const int4*)K_global)[tid];
    }
    if (tid < 2 * nbatch_fa) {
        K_scales[tid] = K_scales_global[tid];
    }
    __syncthreads();

    // Each warp processes 2 columns
    constexpr int cpw = ncols / N_WARPS;  // 2 cols per warp

    for (int jc0 = 0; jc0 < cpw; jc0++) {
        const int jc = jc0 + warp_id * cpw;

        // Hoist Q scales
        half q_scales_hoisted[2];
        q_scales_hoisted[0] = Q_scales[0 * ncols + jc];
        q_scales_hoisted[1] = Q_scales[1 * ncols + jc];

        // Each lane processes one K row
        const int i_KQ = lane_id;
        if (i_KQ >= nbatch_fa) continue;

        const int8_t* K_row = K_values + i_KQ * K_row_stride;
        float acc = 0.0f;

        // Process block 0
        {
            const int4* K_ptr = (const int4*)(K_row + 0);
            const int4* Q_ptr = (const int4*)(Q_values + jc * DKQ + 0);

            int4 K_lo = K_ptr[0];
            int4 K_hi = K_ptr[1];
            int4 Q_lo = Q_ptr[0];
            int4 Q_hi = Q_ptr[1];

            int acc_int = 0;
            acc_int = dp4a(K_lo.x, Q_lo.x, acc_int);
            acc_int = dp4a(K_lo.y, Q_lo.y, acc_int);
            acc_int = dp4a(K_lo.z, Q_lo.z, acc_int);
            acc_int = dp4a(K_lo.w, Q_lo.w, acc_int);
            acc_int = dp4a(K_hi.x, Q_hi.x, acc_int);
            acc_int = dp4a(K_hi.y, Q_hi.y, acc_int);
            acc_int = dp4a(K_hi.z, Q_hi.z, acc_int);
            acc_int = dp4a(K_hi.w, Q_hi.w, acc_int);

            half k_scale = K_scales[0 * nbatch_fa + i_KQ];
            acc += __half2float(__hmul(k_scale, q_scales_hoisted[0])) * (float)acc_int;
        }

        // Process block 1
        {
            const int4* K_ptr = (const int4*)(K_row + 32);
            const int4* Q_ptr = (const int4*)(Q_values + jc * DKQ + 32);

            int4 K_lo = K_ptr[0];
            int4 K_hi = K_ptr[1];
            int4 Q_lo = Q_ptr[0];
            int4 Q_hi = Q_ptr[1];

            int acc_int = 0;
            acc_int = dp4a(K_lo.x, Q_lo.x, acc_int);
            acc_int = dp4a(K_lo.y, Q_lo.y, acc_int);
            acc_int = dp4a(K_lo.z, Q_lo.z, acc_int);
            acc_int = dp4a(K_lo.w, Q_lo.w, acc_int);
            acc_int = dp4a(K_hi.x, Q_hi.x, acc_int);
            acc_int = dp4a(K_hi.y, Q_hi.y, acc_int);
            acc_int = dp4a(K_hi.z, Q_hi.z, acc_int);
            acc_int = dp4a(K_hi.w, Q_hi.w, acc_int);

            half k_scale = K_scales[1 * nbatch_fa + i_KQ];
            acc += __half2float(__hmul(k_scale, q_scales_hoisted[1])) * (float)acc_int;
        }

        KQ_out[i_KQ * ncols + jc] = acc;
    }
}

// OPTIMIZED: Interleaved blocks + prefetched combined scales
__global__ void fattn_kq_optimized(
    const int8_t* __restrict__ K_global,
    const half* __restrict__ K_scales_global,
    const int8_t* __restrict__ Q_values,
    const half* __restrict__ Q_scales,
    float* __restrict__ KQ_out
) {
    __shared__ int8_t K_values[nbatch_fa * K_row_stride];
    __shared__ half K_scales[2 * nbatch_fa];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Load K tile to LDS
    if (tid < nbatch_fa * 32 / 4) {
        ((int4*)K_values)[tid] = ((const int4*)K_global)[tid];
    }
    if (tid < 2 * nbatch_fa) {
        K_scales[tid] = K_scales_global[tid];
    }
    __syncthreads();

    constexpr int cpw = ncols / N_WARPS;

    for (int jc0 = 0; jc0 < cpw; jc0++) {
        const int jc = jc0 + warp_id * cpw;

        // Pre-load Q scales
        half q0_scale = Q_scales[0 * ncols + jc];
        half q1_scale = Q_scales[1 * ncols + jc];

        const int i_KQ = lane_id;
        if (i_KQ >= nbatch_fa) continue;

        // Pre-load K scales and compute combined scales
        half k0_scale = K_scales[0 * nbatch_fa + i_KQ];
        half k1_scale = K_scales[1 * nbatch_fa + i_KQ];
        float scale0 = __half2float(__hmul(k0_scale, q0_scale));
        float scale1 = __half2float(__hmul(k1_scale, q1_scale));

        const int8_t* K_row = K_values + i_KQ * K_row_stride;

        // Load all K and Q data
        const int4* K_ptr0 = (const int4*)(K_row + 0);
        const int4* K_ptr1 = (const int4*)(K_row + 32);
        const int4* Q_ptr0 = (const int4*)(Q_values + jc * DKQ + 0);
        const int4* Q_ptr1 = (const int4*)(Q_values + jc * DKQ + 32);

        int4 K0_lo = K_ptr0[0];
        int4 K0_hi = K_ptr0[1];
        int4 K1_lo = K_ptr1[0];
        int4 K1_hi = K_ptr1[1];
        int4 Q0_lo = Q_ptr0[0];
        int4 Q0_hi = Q_ptr0[1];
        int4 Q1_lo = Q_ptr1[0];
        int4 Q1_hi = Q_ptr1[1];

        // Interleaved dp4a
        int acc0 = 0, acc1 = 0;
        acc0 = dp4a(K0_lo.x, Q0_lo.x, acc0);
        acc1 = dp4a(K1_lo.x, Q1_lo.x, acc1);
        acc0 = dp4a(K0_lo.y, Q0_lo.y, acc0);
        acc1 = dp4a(K1_lo.y, Q1_lo.y, acc1);
        acc0 = dp4a(K0_lo.z, Q0_lo.z, acc0);
        acc1 = dp4a(K1_lo.z, Q1_lo.z, acc1);
        acc0 = dp4a(K0_lo.w, Q0_lo.w, acc0);
        acc1 = dp4a(K1_lo.w, Q1_lo.w, acc1);
        acc0 = dp4a(K0_hi.x, Q0_hi.x, acc0);
        acc1 = dp4a(K1_hi.x, Q1_hi.x, acc1);
        acc0 = dp4a(K0_hi.y, Q0_hi.y, acc0);
        acc1 = dp4a(K1_hi.y, Q1_hi.y, acc1);
        acc0 = dp4a(K0_hi.z, Q0_hi.z, acc0);
        acc1 = dp4a(K1_hi.z, Q1_hi.z, acc1);
        acc0 = dp4a(K0_hi.w, Q0_hi.w, acc0);
        acc1 = dp4a(K1_hi.w, Q1_hi.w, acc1);

        // Use FMA for final accumulation
        float result = __fmaf_rn(scale0, (float)acc0, scale1 * (float)acc1);
        KQ_out[i_KQ * ncols + jc] = result;
    }
}

// OPTIMIZED V2: 4-way interleave
__global__ void fattn_kq_optimized_v2(
    const int8_t* __restrict__ K_global,
    const half* __restrict__ K_scales_global,
    const int8_t* __restrict__ Q_values,
    const half* __restrict__ Q_scales,
    float* __restrict__ KQ_out
) {
    __shared__ int8_t K_values[nbatch_fa * K_row_stride];
    __shared__ half K_scales[2 * nbatch_fa];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (tid < nbatch_fa * 32 / 4) {
        ((int4*)K_values)[tid] = ((const int4*)K_global)[tid];
    }
    if (tid < 2 * nbatch_fa) {
        K_scales[tid] = K_scales_global[tid];
    }
    __syncthreads();

    constexpr int cpw = ncols / N_WARPS;

    for (int jc0 = 0; jc0 < cpw; jc0++) {
        const int jc = jc0 + warp_id * cpw;

        half q0_scale = Q_scales[0 * ncols + jc];
        half q1_scale = Q_scales[1 * ncols + jc];

        const int i_KQ = lane_id;
        if (i_KQ >= nbatch_fa) continue;

        half k0_scale = K_scales[0 * nbatch_fa + i_KQ];
        half k1_scale = K_scales[1 * nbatch_fa + i_KQ];
        float scale0 = __half2float(__hmul(k0_scale, q0_scale));
        float scale1 = __half2float(__hmul(k1_scale, q1_scale));

        const int8_t* K_row = K_values + i_KQ * K_row_stride;

        const int4* K_ptr0 = (const int4*)(K_row + 0);
        const int4* K_ptr1 = (const int4*)(K_row + 32);
        const int4* Q_ptr0 = (const int4*)(Q_values + jc * DKQ + 0);
        const int4* Q_ptr1 = (const int4*)(Q_values + jc * DKQ + 32);

        int4 K0_lo = K_ptr0[0];
        int4 K0_hi = K_ptr0[1];
        int4 K1_lo = K_ptr1[0];
        int4 K1_hi = K_ptr1[1];
        int4 Q0_lo = Q_ptr0[0];
        int4 Q0_hi = Q_ptr0[1];
        int4 Q1_lo = Q_ptr1[0];
        int4 Q1_hi = Q_ptr1[1];

        // 4 independent accumulators
        int acc0a = 0, acc0b = 0, acc1a = 0, acc1b = 0;

        acc0a = dp4a(K0_lo.x, Q0_lo.x, acc0a);
        acc0b = dp4a(K0_hi.x, Q0_hi.x, acc0b);
        acc1a = dp4a(K1_lo.x, Q1_lo.x, acc1a);
        acc1b = dp4a(K1_hi.x, Q1_hi.x, acc1b);

        acc0a = dp4a(K0_lo.y, Q0_lo.y, acc0a);
        acc0b = dp4a(K0_hi.y, Q0_hi.y, acc0b);
        acc1a = dp4a(K1_lo.y, Q1_lo.y, acc1a);
        acc1b = dp4a(K1_hi.y, Q1_hi.y, acc1b);

        acc0a = dp4a(K0_lo.z, Q0_lo.z, acc0a);
        acc0b = dp4a(K0_hi.z, Q0_hi.z, acc0b);
        acc1a = dp4a(K1_lo.z, Q1_lo.z, acc1a);
        acc1b = dp4a(K1_hi.z, Q1_hi.z, acc1b);

        acc0a = dp4a(K0_lo.w, Q0_lo.w, acc0a);
        acc0b = dp4a(K0_hi.w, Q0_hi.y, acc0b);
        acc1a = dp4a(K1_lo.w, Q1_lo.w, acc1a);
        acc1b = dp4a(K1_hi.w, Q1_hi.w, acc1b);

        int acc0 = acc0a + acc0b;
        int acc1 = acc1a + acc1b;

        float result = __fmaf_rn(scale0, (float)acc0, scale1 * (float)acc1);
        KQ_out[i_KQ * ncols + jc] = result;
    }
}

int main() {
    printf("Testing K·Q dp4a with LDS (flash attention pattern)\n");
    printf("nbatch_fa=%d, ncols=%d, DKQ=%d, K_row_stride=%d\n\n",
           nbatch_fa, ncols, DKQ, K_row_stride);

    // Allocate
    int8_t *d_K_global, *d_Q;
    half *d_K_scales, *d_Q_scales;
    float *d_out;

    CHECK_HIP(hipMalloc(&d_K_global, nbatch_fa * 64));
    CHECK_HIP(hipMalloc(&d_Q, ncols * DKQ));
    CHECK_HIP(hipMalloc(&d_K_scales, 2 * nbatch_fa * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_Q_scales, 2 * ncols * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_out, nbatch_fa * ncols * sizeof(float)));

    // Initialize
    {
        int8_t* h_K = new int8_t[nbatch_fa * 64];
        int8_t* h_Q = new int8_t[ncols * DKQ];
        half* h_K_scales = new half[2 * nbatch_fa];
        half* h_Q_scales = new half[2 * ncols];

        for (int i = 0; i < nbatch_fa * 64; i++) h_K[i] = rand() % 256 - 128;
        for (int i = 0; i < ncols * DKQ; i++) h_Q[i] = rand() % 256 - 128;
        for (int i = 0; i < 2 * nbatch_fa; i++) h_K_scales[i] = __float2half(0.01f * (rand() % 100));
        for (int i = 0; i < 2 * ncols; i++) h_Q_scales[i] = __float2half(0.01f * (rand() % 100));

        CHECK_HIP(hipMemcpy(d_K_global, h_K, nbatch_fa * 64, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_Q, h_Q, ncols * DKQ, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_K_scales, h_K_scales, 2 * nbatch_fa * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_Q_scales, h_Q_scales, 2 * ncols * sizeof(half), hipMemcpyHostToDevice));

        delete[] h_K;
        delete[] h_Q;
        delete[] h_K_scales;
        delete[] h_Q_scales;
    }

    // Run many iterations to simulate actual workload
    const int n_tiles = 100;  // Simulate processing many K tiles
    const int warmup = 50;
    const int iters = 500;

    struct {
        const char* name;
        void (*kernel)(const int8_t*, const half*, const int8_t*, const half*, float*);
    } variants[] = {
        {"baseline", fattn_kq_baseline},
        {"optimized (2-interleave)", fattn_kq_optimized},
        {"optimized_v2 (4-interleave)", fattn_kq_optimized_v2},
    };

    double baseline_time = 0;

    for (int v = 0; v < 3; v++) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            for (int t = 0; t < n_tiles; t++) {
                variants[v].kernel<<<1, BLOCK_SIZE>>>(d_K_global, d_K_scales, d_Q, d_Q_scales, d_out);
            }
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            for (int t = 0; t < n_tiles; t++) {
                variants[v].kernel<<<1, BLOCK_SIZE>>>(d_K_global, d_K_scales, d_Q, d_Q_scales, d_out);
            }
        }
        CHECK_HIP(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double time_us = std::chrono::duration<double, std::micro>(end - start).count() / iters;

        if (v == 0) baseline_time = time_us;

        printf("%-30s: %8.2f us  (%.2fx baseline)\n",
               variants[v].name, time_us, baseline_time / time_us);
    }

    CHECK_HIP(hipFree(d_K_global));
    CHECK_HIP(hipFree(d_Q));
    CHECK_HIP(hipFree(d_K_scales));
    CHECK_HIP(hipFree(d_Q_scales));
    CHECK_HIP(hipFree(d_out));

    return 0;
}
