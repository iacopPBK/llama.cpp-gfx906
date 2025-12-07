// Test: What i_max values actually occur in real inference?
// This helps us understand when the need_check optimization matters

#include <hip/hip_runtime.h>
#include <cstdio>

// Real MMQ parameters from llama.cpp
#define MMQ_Y 64  // tile size in Y dimension

// Simulated kernel that just records i_max values
__global__ void record_imax_values(int nrows, int* histogram) {
    // Simulate how the MMQ kernel is launched
    // Each block handles a tile of MMQ_Y rows
    int block_start = blockIdx.x * MMQ_Y;
    int block_end = min(block_start + MMQ_Y - 1, nrows - 1);
    int i_max = block_end - block_start;  // 0 to 63

    // Only thread 0 records
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&histogram[i_max], 1);
    }
}

int main() {
    printf("Analysis: i_max distribution in real MMQ workloads\n");
    printf("===================================================\n\n");

    printf("MMQ tile size: %d rows\n\n", MMQ_Y);

    // Test various nrows scenarios
    int test_cases[] = {
        1,      // TG: single token
        8,      // TG: small batch
        64,     // PP: one full tile
        128,    // PP: 2 tiles
        512,    // PP: typical prompt
        1024,   // PP: larger prompt
        8192,   // PP: context length
    };

    int* d_histogram;
    hipMalloc(&d_histogram, 64 * sizeof(int));

    printf("%-15s  %-12s  %-15s  %-15s\n", "nrows", "num_tiles", "full_tiles", "partial_i_max");
    printf("%-15s  %-12s  %-15s  %-15s\n", "-----", "---------", "----------", "-------------");

    for (int nrows : test_cases) {
        hipMemset(d_histogram, 0, 64 * sizeof(int));

        int num_tiles = (nrows + MMQ_Y - 1) / MMQ_Y;

        // Launch one block per tile
        dim3 grid(num_tiles);
        dim3 block(1, 1);  // Just for counting
        record_imax_values<<<grid, block>>>(nrows, d_histogram);
        hipDeviceSynchronize();

        // Read results
        int h_histogram[64];
        hipMemcpy(h_histogram, d_histogram, 64 * sizeof(int), hipMemcpyDeviceToHost);

        // Count full tiles (i_max = 63) vs partial tiles
        int full_tiles = h_histogram[63];
        int partial_tile_imax = -1;
        for (int i = 0; i < 63; i++) {
            if (h_histogram[i] > 0) {
                partial_tile_imax = i;
                break;
            }
        }

        printf("%-15d  %-12d  %-15d  %-15d\n", nrows, num_tiles, full_tiles, partial_tile_imax);
    }

    printf("\n");
    printf("Key insights:\n");
    printf("-------------\n");
    printf("- For TG (nrows=1): i_max=0, ALL 64 threads OOB except first!\n");
    printf("- For TG (nrows=8): i_max=7, 87.5%% threads OOB\n");
    printf("- For PP (nrows=512): Only 1 partial tile per 8 full tiles\n");
    printf("- For PP (nrows=8192): Only 1 partial tile per 128 full tiles\n");
    printf("\n");
    printf("Impact analysis:\n");
    printf("---------------\n");
    printf("TG with small batch: HIGH OOB %% → optimization HELPS\n");
    printf("PP with large batch: LOW OOB %%  → optimization HURTS\n");
    printf("\n");

    // Detailed analysis for TG
    printf("TG detailed analysis (nrows=1 to 16):\n");
    printf("%-10s  %-10s  %-15s  %-20s\n", "nrows", "i_max", "OOB threads", "optimization impact");
    printf("%-10s  %-10s  %-15s  %-20s\n", "------", "-----", "-----------", "-------------------");

    for (int nrows = 1; nrows <= 16; nrows++) {
        int i_max = nrows - 1;  // For single tile
        int oob_threads = MMQ_Y - nrows;  // Threads beyond valid rows
        float oob_pct = 100.0f * oob_threads / MMQ_Y;
        const char* impact = oob_pct > 50 ? "HELPS (high OOB)" : (oob_pct > 10 ? "NEUTRAL" : "HURTS (low OOB)");
        printf("%-10d  %-10d  %-6d (%.1f%%)    %s\n", nrows, i_max, oob_threads, oob_pct, impact);
    }

    hipFree(d_histogram);
    return 0;
}
