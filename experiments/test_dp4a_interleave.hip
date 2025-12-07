// Test: Breaking dp4a dependency chain via interleaving
// GFX906 dp4a (v_dot4_i32_i8) has ~8 cycle latency
// Current: 8 chained dp4a = 64 cycles of latency per block
// Goal: Interleave 2 blocks to hide latency

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

// dp4a intrinsic
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// BASELINE: Sequential blocks (current implementation)
__global__ void kq_baseline(
    const int4* __restrict__ K_values,  // [batch, 2 blocks * 2 int4]
    const int4* __restrict__ Q_values,  // [ncols, 2 blocks * 2 int4]
    const half* __restrict__ K_scales,  // [2 blocks, batch]
    const half* __restrict__ Q_scales,  // [2 blocks, ncols]
    float* __restrict__ KQ_out,         // [batch, ncols]
    int batch, int ncols
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Each thread handles one (batch_row, col) pair
    const int row = bid * 64 + tid;
    if (row >= batch) return;

    for (int col = 0; col < ncols; col++) {
        float acc = 0.0f;

        // Process block 0
        {
            const int4* K_ptr = K_values + row * 4 + 0;  // 2 int4 for block 0
            const int4* Q_ptr = Q_values + col * 4 + 0;

            int4 K_lo = K_ptr[0];
            int4 K_hi = K_ptr[1];
            int4 Q_lo = Q_ptr[0];
            int4 Q_hi = Q_ptr[1];

            // 8 chained dp4a
            int acc_int = 0;
            acc_int = dp4a(K_lo.x, Q_lo.x, acc_int);
            acc_int = dp4a(K_lo.y, Q_lo.y, acc_int);
            acc_int = dp4a(K_lo.z, Q_lo.z, acc_int);
            acc_int = dp4a(K_lo.w, Q_lo.w, acc_int);
            acc_int = dp4a(K_hi.x, Q_hi.x, acc_int);
            acc_int = dp4a(K_hi.y, Q_hi.y, acc_int);
            acc_int = dp4a(K_hi.z, Q_hi.z, acc_int);
            acc_int = dp4a(K_hi.w, Q_hi.w, acc_int);

            half k_scale = K_scales[0 * batch + row];
            half q_scale = Q_scales[0 * ncols + col];
            acc += __half2float(__hmul(k_scale, q_scale)) * (float)acc_int;
        }

        // Process block 1
        {
            const int4* K_ptr = K_values + row * 4 + 2;  // 2 int4 for block 1
            const int4* Q_ptr = Q_values + col * 4 + 2;

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

            half k_scale = K_scales[1 * batch + row];
            half q_scale = Q_scales[1 * ncols + col];
            acc += __half2float(__hmul(k_scale, q_scale)) * (float)acc_int;
        }

        KQ_out[row * ncols + col] = acc;
    }
}

// INTERLEAVED: Process both blocks simultaneously to hide dp4a latency
__global__ void kq_interleaved(
    const int4* __restrict__ K_values,
    const int4* __restrict__ Q_values,
    const half* __restrict__ K_scales,
    const half* __restrict__ Q_scales,
    float* __restrict__ KQ_out,
    int batch, int ncols
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int row = bid * 64 + tid;
    if (row >= batch) return;

    for (int col = 0; col < ncols; col++) {
        // Load all data upfront
        const int4* K_ptr0 = K_values + row * 4 + 0;
        const int4* K_ptr1 = K_values + row * 4 + 2;
        const int4* Q_ptr0 = Q_values + col * 4 + 0;
        const int4* Q_ptr1 = Q_values + col * 4 + 2;

        int4 K0_lo = K_ptr0[0];
        int4 K0_hi = K_ptr0[1];
        int4 K1_lo = K_ptr1[0];
        int4 K1_hi = K_ptr1[1];
        int4 Q0_lo = Q_ptr0[0];
        int4 Q0_hi = Q_ptr0[1];
        int4 Q1_lo = Q_ptr1[0];
        int4 Q1_hi = Q_ptr1[1];

        // Interleave dp4a from both blocks
        // This allows the compiler to schedule instructions to hide latency
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

        // Dequantize
        half k0_scale = K_scales[0 * batch + row];
        half k1_scale = K_scales[1 * batch + row];
        half q0_scale = Q_scales[0 * ncols + col];
        half q1_scale = Q_scales[1 * ncols + col];

        float result = __half2float(__hmul(k0_scale, q0_scale)) * (float)acc0 +
                       __half2float(__hmul(k1_scale, q1_scale)) * (float)acc1;

        KQ_out[row * ncols + col] = result;
    }
}

// INTERLEAVED V2: Use 4 accumulators (unroll both blocks into 4 parts)
__global__ void kq_interleaved_v2(
    const int4* __restrict__ K_values,
    const int4* __restrict__ Q_values,
    const half* __restrict__ K_scales,
    const half* __restrict__ Q_scales,
    float* __restrict__ KQ_out,
    int batch, int ncols
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int row = bid * 64 + tid;
    if (row >= batch) return;

    for (int col = 0; col < ncols; col++) {
        const int4* K_ptr0 = K_values + row * 4 + 0;
        const int4* K_ptr1 = K_values + row * 4 + 2;
        const int4* Q_ptr0 = Q_values + col * 4 + 0;
        const int4* Q_ptr1 = Q_values + col * 4 + 2;

        int4 K0_lo = K_ptr0[0];
        int4 K0_hi = K_ptr0[1];
        int4 K1_lo = K_ptr1[0];
        int4 K1_hi = K_ptr1[1];
        int4 Q0_lo = Q_ptr0[0];
        int4 Q0_hi = Q_ptr0[1];
        int4 Q1_lo = Q_ptr1[0];
        int4 Q1_hi = Q_ptr1[1];

        // 4 independent accumulators: block0_lo, block0_hi, block1_lo, block1_hi
        int acc0a = 0, acc0b = 0, acc1a = 0, acc1b = 0;

        // Round 1: .x components
        acc0a = dp4a(K0_lo.x, Q0_lo.x, acc0a);
        acc0b = dp4a(K0_hi.x, Q0_hi.x, acc0b);
        acc1a = dp4a(K1_lo.x, Q1_lo.x, acc1a);
        acc1b = dp4a(K1_hi.x, Q1_hi.x, acc1b);

        // Round 2: .y components
        acc0a = dp4a(K0_lo.y, Q0_lo.y, acc0a);
        acc0b = dp4a(K0_hi.y, Q0_hi.y, acc0b);
        acc1a = dp4a(K1_lo.y, Q1_lo.y, acc1a);
        acc1b = dp4a(K1_hi.y, Q1_hi.y, acc1b);

        // Round 3: .z components
        acc0a = dp4a(K0_lo.z, Q0_lo.z, acc0a);
        acc0b = dp4a(K0_hi.z, Q0_hi.z, acc0b);
        acc1a = dp4a(K1_lo.z, Q1_lo.z, acc1a);
        acc1b = dp4a(K1_hi.z, Q1_hi.z, acc1b);

        // Round 4: .w components
        acc0a = dp4a(K0_lo.w, Q0_lo.w, acc0a);
        acc0b = dp4a(K0_hi.w, Q0_hi.w, acc0b);
        acc1a = dp4a(K1_lo.w, Q1_lo.w, acc1a);
        acc1b = dp4a(K1_hi.w, Q1_hi.w, acc1b);

        // Combine accumulators
        int acc0 = acc0a + acc0b;
        int acc1 = acc1a + acc1b;

        half k0_scale = K_scales[0 * batch + row];
        half k1_scale = K_scales[1 * batch + row];
        half q0_scale = Q_scales[0 * ncols + col];
        half q1_scale = Q_scales[1 * ncols + col];

        float result = __half2float(__hmul(k0_scale, q0_scale)) * (float)acc0 +
                       __half2float(__hmul(k1_scale, q1_scale)) * (float)acc1;

        KQ_out[row * ncols + col] = result;
    }
}

// PREFETCH SCALES: Load scales early to overlap with dp4a computation
__global__ void kq_prefetch_scales(
    const int4* __restrict__ K_values,
    const int4* __restrict__ Q_values,
    const half* __restrict__ K_scales,
    const half* __restrict__ Q_scales,
    float* __restrict__ KQ_out,
    int batch, int ncols
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int row = bid * 64 + tid;
    if (row >= batch) return;

    // Prefetch K scales (common across all cols)
    half k0_scale = K_scales[0 * batch + row];
    half k1_scale = K_scales[1 * batch + row];

    for (int col = 0; col < ncols; col++) {
        // Prefetch Q scales
        half q0_scale = Q_scales[0 * ncols + col];
        half q1_scale = Q_scales[1 * ncols + col];
        float scale0 = __half2float(__hmul(k0_scale, q0_scale));
        float scale1 = __half2float(__hmul(k1_scale, q1_scale));

        const int4* K_ptr0 = K_values + row * 4 + 0;
        const int4* K_ptr1 = K_values + row * 4 + 2;
        const int4* Q_ptr0 = Q_values + col * 4 + 0;
        const int4* Q_ptr1 = Q_values + col * 4 + 2;

        int4 K0_lo = K_ptr0[0];
        int4 K0_hi = K_ptr0[1];
        int4 K1_lo = K_ptr1[0];
        int4 K1_hi = K_ptr1[1];
        int4 Q0_lo = Q_ptr0[0];
        int4 Q0_hi = Q_ptr0[1];
        int4 Q1_lo = Q_ptr1[0];
        int4 Q1_hi = Q_ptr1[1];

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

        KQ_out[row * ncols + col] = scale0 * (float)acc0 + scale1 * (float)acc1;
    }
}

int main() {
    // Simulate flash attention dimensions
    const int batch = 2048;     // K rows to process
    const int ncols = 8;        // Q columns (ncols in fattn)
    const int DKQ = 64;         // dimension (2 blocks of 32)

    printf("Testing K·Q dp4a optimization strategies\n");
    printf("batch=%d, ncols=%d, DKQ=%d\n\n", batch, ncols, DKQ);

    // Allocate
    int4 *d_K, *d_Q;
    half *d_K_scales, *d_Q_scales;
    float *d_out;

    // K: [batch, 4 int4] = [batch, 2 blocks * 2 int4 per block]
    CHECK_HIP(hipMalloc(&d_K, batch * 4 * sizeof(int4)));
    CHECK_HIP(hipMalloc(&d_Q, ncols * 4 * sizeof(int4)));
    CHECK_HIP(hipMalloc(&d_K_scales, 2 * batch * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_Q_scales, 2 * ncols * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_out, batch * ncols * sizeof(float)));

    // Initialize with random data
    {
        int4* h_K = new int4[batch * 4];
        int4* h_Q = new int4[ncols * 4];
        half* h_K_scales = new half[2 * batch];
        half* h_Q_scales = new half[2 * ncols];

        for (int i = 0; i < batch * 4; i++) {
            h_K[i] = make_int4(rand(), rand(), rand(), rand());
        }
        for (int i = 0; i < ncols * 4; i++) {
            h_Q[i] = make_int4(rand(), rand(), rand(), rand());
        }
        for (int i = 0; i < 2 * batch; i++) {
            h_K_scales[i] = __float2half(0.01f * (rand() % 100));
        }
        for (int i = 0; i < 2 * ncols; i++) {
            h_Q_scales[i] = __float2half(0.01f * (rand() % 100));
        }

        CHECK_HIP(hipMemcpy(d_K, h_K, batch * 4 * sizeof(int4), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_Q, h_Q, ncols * 4 * sizeof(int4), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_K_scales, h_K_scales, 2 * batch * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_Q_scales, h_Q_scales, 2 * ncols * sizeof(half), hipMemcpyHostToDevice));

        delete[] h_K;
        delete[] h_Q;
        delete[] h_K_scales;
        delete[] h_Q_scales;
    }

    dim3 grid((batch + 63) / 64);
    dim3 block(64);

    const int warmup = 100;
    const int iters = 1000;

    // Benchmark each variant
    struct {
        const char* name;
        void (*kernel)(const int4*, const int4*, const half*, const half*, float*, int, int);
    } variants[] = {
        {"baseline (sequential)", kq_baseline},
        {"interleaved (2 acc)", kq_interleaved},
        {"interleaved_v2 (4 acc)", kq_interleaved_v2},
        {"prefetch_scales", kq_prefetch_scales},
    };

    double baseline_time = 0;

    for (int v = 0; v < 4; v++) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            variants[v].kernel<<<grid, block>>>(d_K, d_Q, d_K_scales, d_Q_scales, d_out, batch, ncols);
        }
        CHECK_HIP(hipDeviceSynchronize());

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            variants[v].kernel<<<grid, block>>>(d_K, d_Q, d_K_scales, d_Q_scales, d_out, batch, ncols);
        }
        CHECK_HIP(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double time_us = std::chrono::duration<double, std::micro>(end - start).count() / iters;

        if (v == 0) baseline_time = time_us;

        printf("%-25s: %8.2f us  (%.2fx baseline)\n",
               variants[v].name, time_us, baseline_time / time_us);
    }

    CHECK_HIP(hipFree(d_K));
    CHECK_HIP(hipFree(d_Q));
    CHECK_HIP(hipFree(d_K_scales));
    CHECK_HIP(hipFree(d_Q_scales));
    CHECK_HIP(hipFree(d_out));

    return 0;
}
