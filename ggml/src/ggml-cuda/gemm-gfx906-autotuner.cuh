#pragma once

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>

// GFX906 Hardware Constraints (from ISA documentation)
namespace GFX906 {
    constexpr int MAX_WAVES_PER_CU = 40;
    constexpr int WAVE_SIZE = 64;
    constexpr int MAX_THREADS_PER_CU = MAX_WAVES_PER_CU * WAVE_SIZE; // 2560
    
    // VGPR constraints: 256 VGPRs available per thread
    constexpr int VGPRS_PER_THREAD_MAX = 256;
    constexpr int SIMDS_PER_CU = 4;
    constexpr int MAX_WAVES_PER_SIMD = 10;  // 40 waves / 4 SIMDs
    
    // SGPR constraints: 800 SGPRs per CU, allocated in groups of 16
    constexpr int SGPRS_PER_WAVE_MAX = 102;  // Practical limit
    constexpr int SGPR_ALLOCATION_GRANULARITY = 16;
    
    constexpr int LDS_SIZE = 65536; // 64 KB
    constexpr int LDS_GRANULARITY = 512; // bytes (128 dwords)
    
    constexpr int L1_CACHE_SIZE = 16384; // 16 KB per CU
    constexpr int L2_CACHE_SIZE = 4194304; // 4 MB total
    constexpr int CACHE_LINE_SIZE = 64; // bytes
    
    constexpr int LDS_BANKS = 32;
    constexpr int LDS_BANK_WIDTH = 4; // bytes
    constexpr int LDS_BANDWIDTH_PER_CYCLE = LDS_BANKS * LDS_BANK_WIDTH; // 128 bytes/cycle
}

// Configuration structure for GEMM kernel
struct GemmConfig {
    // Block and grid configuration
    int threads_per_block;
    int waves_per_block;
    
    // Tile dimensions
    int tile_m;
    int tile_n;
    int tile_k;
    
    // Thread tile dimensions
    int thread_tile_m;
    int thread_tile_n;
    
    // Memory optimization parameters
    int vector_width;     // 1, 2, 4, or 8 (for float/half)
    int lds_pad;          // 0, 1, 2, 4 (to avoid bank conflicts)
    bool double_buffer;   // Use double buffering
    bool use_dot2;        // Use V_DOT2_F32_F16 for FP16
    
    // Calculated resource usage
    int lds_bytes;
    int vgprs_per_thread;
    int sgprs_per_thread;
    int occupancy_waves;
    
    // Performance metrics
    float tflops;
    float time_ms;
    float efficiency;
    
    // Calculate resource usage
    void calculate_resources(bool is_fp16 = false) {
        int elem_size = is_fp16 ? 2 : 4;
        
        // LDS usage calculation
        int lds_a = tile_m * (tile_k + lds_pad) * elem_size;
        int lds_b = tile_k * (tile_n + lds_pad) * elem_size;
        lds_bytes = double_buffer ? 2 * (lds_a + lds_b) : (lds_a + lds_b);
        
        // Round up to LDS granularity
        lds_bytes = ((lds_bytes + GFX906::LDS_GRANULARITY - 1) / GFX906::LDS_GRANULARITY) 
                    * GFX906::LDS_GRANULARITY;
        
        // VGPR estimation (accumulator registers + working registers)
        vgprs_per_thread = thread_tile_m * thread_tile_n  // Accumulators
                          + 2 * std::max(thread_tile_m, thread_tile_n)  // A and B registers
                          + 8;  // Working registers
        
        // SGPR estimation (block indices, loop counters, etc.)
        sgprs_per_thread = 16;
        
        // Calculate occupancy
        calculate_occupancy();
    }
    
    void calculate_occupancy() {
        // Occupancy limited by LDS
        int blocks_lds = GFX906::LDS_SIZE / lds_bytes;
        int waves_lds = blocks_lds * waves_per_block;
        
        // Occupancy limited by VGPRs
        // Each SIMD can run floor(256 / vgprs_per_thread) waves
        int waves_per_simd = GFX906::VGPRS_PER_THREAD_MAX / vgprs_per_thread;
        waves_per_simd = std::min(waves_per_simd, GFX906::MAX_WAVES_PER_SIMD);
        int waves_vgpr = waves_per_simd * GFX906::SIMDS_PER_CU;
        
        // Occupancy limited by SGPRs
        // SGPRs are allocated per wave in multiples of 16
        int sgprs_allocated = ((sgprs_per_thread + 15) / 16) * 16;
        int waves_sgpr = (GFX906::SGPRS_PER_WAVE_MAX * GFX906::MAX_WAVES_PER_CU) / 
                         (sgprs_allocated * waves_per_block);
        
        // Take minimum of all limits
        occupancy_waves = std::min({waves_lds, waves_vgpr, waves_sgpr, GFX906::MAX_WAVES_PER_CU});
    }
    
    bool is_valid() const {
        // Check if configuration fits within hardware limits
        return lds_bytes <= GFX906::LDS_SIZE &&
               vgprs_per_thread <= 256 &&  // Max VGPRs per thread
               sgprs_per_thread <= 104 &&  // Max SGPRs per wave (typical limit)
               occupancy_waves > 0;
    }
};

// Auto-tuner class
class GemmAutoTuner {
private:
    // Cache for optimal configurations
    std::map<std::tuple<int, int, int, bool>, GemmConfig> config_cache;
    
    // Generate candidate configurations based on matrix dimensions
    std::vector<GemmConfig> generate_candidates(int M, int N, int K, bool is_fp16) {
        std::vector<GemmConfig> configs;
        
        // Determine reasonable ranges based on matrix size
        int min_tile = std::min({M, N, K, 32});
        int max_tile = std::min({M, N, K, 256});
        
        // Thread configurations (must be multiple of wave size)
        std::vector<int> thread_counts = {64, 128, 256};
        if (M * N < 65536) thread_counts.push_back(64);  // Small matrices
        
        // Tile K values (affects register pressure)
        std::vector<int> tile_k_values = {16, 32};
        if (K >= 64) tile_k_values.push_back(64);
        
        // Thread tile sizes
        std::vector<std::pair<int, int>> thread_tiles;
        if (is_fp16) {
            thread_tiles = {{2, 2}, {2, 4}, {4, 2}, {4, 4}};
        } else {
            thread_tiles = {{2, 2}, {4, 4}, {8, 8}};
        }
        
        // Generate configurations
        for (int threads : thread_counts) {
            for (int tile_k : tile_k_values) {
                // Calculate maximum tile_m and tile_n based on LDS constraint
                int elem_size = is_fp16 ? 2 : 4;
                int max_lds = GFX906::LDS_SIZE;
                
                // Try different tile dimensions
                for (int tile_m = min_tile; tile_m <= max_tile; tile_m *= 2) {
                    for (int tile_n = min_tile; tile_n <= max_tile; tile_n *= 2) {
                        // Skip if tiles don't evenly divide matrix dimensions
                        if (M % tile_m != 0 || N % tile_n != 0) continue;
                        
                        // Try different thread tile configurations
                        for (auto [tt_m, tt_n] : thread_tiles) {
                            // Check if thread tiles divide tile dimensions
                            if (tile_m % tt_m != 0 || tile_n % tt_n != 0) continue;
                            
                            // Check if we have enough threads
                            int threads_needed = (tile_m / tt_m) * (tile_n / tt_n);
                            if (threads_needed > threads) continue;
                            
                            // Try different memory optimizations
                            for (int pad : {0, 1}) {
                                for (bool use_double : {false, true}) {
                                    GemmConfig config;
                                    config.threads_per_block = threads;
                                    config.waves_per_block = threads / GFX906::WAVE_SIZE;
                                    config.tile_m = tile_m;
                                    config.tile_n = tile_n;
                                    config.tile_k = tile_k;
                                    config.thread_tile_m = tt_m;
                                    config.thread_tile_n = tt_n;
                                    config.vector_width = is_fp16 ? 2 : 4;
                                    config.lds_pad = pad;
                                    config.double_buffer = use_double;
                                    config.use_dot2 = is_fp16;
                                    
                                    config.calculate_resources(is_fp16);
                                    
                                    if (config.is_valid()) {
                                        configs.push_back(config);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by expected performance (heuristic)
        std::sort(configs.begin(), configs.end(), [](const GemmConfig& a, const GemmConfig& b) {
            // Prefer higher occupancy
            if (a.occupancy_waves != b.occupancy_waves)
                return a.occupancy_waves > b.occupancy_waves;
            
            // Prefer larger tiles (better data reuse)
            int a_tile_size = a.tile_m * a.tile_n;
            int b_tile_size = b.tile_m * b.tile_n;
            if (a_tile_size != b_tile_size)
                return a_tile_size > b_tile_size;
            
            // Prefer configurations that use special instructions
            return a.use_dot2 > b.use_dot2;
        });
        
        // Return top candidates
        if (configs.size() > 20) {
            configs.resize(20);
        }
        
        return configs;
    }
    
    // Benchmark a single configuration
    float benchmark_config(const GemmConfig& config, 
                          const void* A, const void* B, void* C,
                          int M, int N, int K, bool is_fp16,
                          hipStream_t stream = nullptr) {
        // This would launch the actual kernel with the given configuration
        // For now, return a placeholder
        return 0.0f;
    }
    
public:
    // Main tuning function
    GemmConfig tune(int M, int N, int K, bool is_fp16 = false,
                   const void* A = nullptr, const void* B = nullptr, void* C = nullptr) {
        
        auto key = std::make_tuple(M, N, K, is_fp16);
        
        // Check cache first
        auto it = config_cache.find(key);
        if (it != config_cache.end()) {
            return it->second;
        }
        
        // Generate candidate configurations
        std::vector<GemmConfig> candidates = generate_candidates(M, N, K, is_fp16);
        
        if (candidates.empty()) {
            // Return default configuration if no valid candidates
            GemmConfig default_config;
            default_config.threads_per_block = 256;
            default_config.tile_m = 64;
            default_config.tile_n = 64;
            default_config.tile_k = 32;
            default_config.thread_tile_m = 2;
            default_config.thread_tile_n = 2;
            return default_config;
        }
        
        // If we have actual data, benchmark all candidates
        GemmConfig best = candidates[0];
        if (A != nullptr && B != nullptr && C != nullptr) {
            float best_time = std::numeric_limits<float>::max();
            
            for (const auto& config : candidates) {
                float time = benchmark_config(config, A, B, C, M, N, K, is_fp16);
                if (time < best_time) {
                    best_time = time;
                    best = config;
                    best.time_ms = time;
                    
                    // Calculate TFLOPS
                    double ops = 2.0 * M * N * K;
                    best.tflops = (ops / 1e12) / (time / 1000.0);
                    
                    // Calculate efficiency
                    double peak_tflops = is_fp16 ? 26.3 : 13.1;
                    best.efficiency = (best.tflops / peak_tflops) * 100.0;
                }
            }
        }
        
        // Cache the result
        config_cache[key] = best;
        
        return best;
    }
    
    // Get occupancy information for a configuration
    void print_occupancy_info(const GemmConfig& config) {
        printf("Occupancy Analysis:\n");
        printf("  Threads per block: %d\n", config.threads_per_block);
        printf("  Waves per block: %d\n", config.waves_per_block);
        printf("  LDS usage: %d bytes (%.1f%% of 64KB)\n", 
               config.lds_bytes, (config.lds_bytes * 100.0) / GFX906::LDS_SIZE);
        printf("  VGPRs per thread: %d\n", config.vgprs_per_thread);
        printf("  SGPRs per thread: %d\n", config.sgprs_per_thread);
        printf("  Occupancy: %d waves (%.1f%% of max 40)\n", 
               config.occupancy_waves, (config.occupancy_waves * 100.0) / GFX906::MAX_WAVES_PER_CU);
        
        // Calculate limiting factor
        int blocks_lds = GFX906::LDS_SIZE / config.lds_bytes;
        int waves_lds = blocks_lds * config.waves_per_block;
        
        int waves_per_simd = GFX906::VGPRS_PER_THREAD_MAX / config.vgprs_per_thread;
        waves_per_simd = std::min(waves_per_simd, GFX906::MAX_WAVES_PER_SIMD);
        int waves_vgpr = waves_per_simd * GFX906::SIMDS_PER_CU;
        
        int sgprs_allocated = ((config.sgprs_per_thread + 15) / 16) * 16;
        int waves_sgpr = (GFX906::SGPRS_PER_WAVE_MAX * GFX906::MAX_WAVES_PER_CU) / 
                         (sgprs_allocated * config.waves_per_block);
        
        printf("  Limiting factor: ");
        if (config.occupancy_waves == waves_lds) {
            printf("LDS (%d blocks * %d waves/block = %d waves)\n", 
                   blocks_lds, config.waves_per_block, waves_lds);
        } else if (config.occupancy_waves == waves_vgpr) {
            printf("VGPRs (%d waves)\n", waves_vgpr);
        } else if (config.occupancy_waves == waves_sgpr) {
            printf("SGPRs (%d waves)\n", waves_sgpr);
        } else {
            printf("Max waves per CU (%d)\n", GFX906::MAX_WAVES_PER_CU);
        }
    }
    
    // Suggest optimizations based on current configuration
    void suggest_optimizations(const GemmConfig& config, int M, int N, int K) {
        printf("\nOptimization Suggestions:\n");
        
        // Check occupancy
        if (config.occupancy_waves < 20) {
            printf("  ⚠ Low occupancy (%d waves). Consider:\n", config.occupancy_waves);
            
            if (config.lds_bytes > GFX906::LDS_SIZE / 2) {
                printf("    - Reducing tile size (current: %dx%dx%d)\n", 
                       config.tile_m, config.tile_n, config.tile_k);
            }
            
            if (config.vgprs_per_thread > 128) {
                printf("    - Reducing thread tile size (current: %dx%d)\n",
                       config.thread_tile_m, config.thread_tile_n);
            }
        }
        
        // Check tile efficiency
        if (M % config.tile_m != 0 || N % config.tile_n != 0) {
            printf("  ⚠ Tiles don't evenly divide matrix. Wasted computation at boundaries.\n");
        }
        
        // Check memory access pattern
        int bytes_per_thread = config.vector_width * (config.thread_tile_m + config.thread_tile_n);
        if (bytes_per_thread < 16) {
            printf("  ⚠ Low memory throughput. Consider increasing vector width or thread tiles.\n");
        }
        
        // Check for bank conflicts
        if (config.lds_pad == 0 && config.tile_k % 32 == 0) {
            printf("  ⚠ Potential LDS bank conflicts. Consider padding (current: %d)\n", config.lds_pad);
        }
    }
};

// Helper to select best configuration for common sizes
class GemmConfigSelector {
private:
    GemmAutoTuner tuner;
    
    // Pre-tuned configurations for common sizes
    std::map<std::tuple<int, int, int>, GemmConfig> pretuned_configs;
    
public:
    GemmConfigSelector() {
        // Initialize with known good configurations for common sizes
        init_pretuned_configs();
    }
    
    void init_pretuned_configs() {
        // Add pre-tuned configurations for common matrix sizes
        // These would be determined through extensive benchmarking
        
        // Example: 1024x1024x1024 FP16
        GemmConfig config_1k_fp16;
        config_1k_fp16.threads_per_block = 256;
        config_1k_fp16.tile_m = 128;
        config_1k_fp16.tile_n = 128;
        config_1k_fp16.tile_k = 32;
        config_1k_fp16.thread_tile_m = 4;
        config_1k_fp16.thread_tile_n = 4;
        config_1k_fp16.vector_width = 2;
        config_1k_fp16.lds_pad = 1;
        config_1k_fp16.double_buffer = false;
        config_1k_fp16.use_dot2 = true;
        config_1k_fp16.calculate_resources(true);
        pretuned_configs[{1024, 1024, 1024}] = config_1k_fp16;
        
        // Add more pre-tuned configs...
    }
    
    GemmConfig select(int M, int N, int K, bool is_fp16 = false) {
        // Check pre-tuned first
        auto key = std::make_tuple(M, N, K);
        auto it = pretuned_configs.find(key);
        if (it != pretuned_configs.end()) {
            return it->second;
        }
        
        // Fall back to auto-tuning
        return tuner.tune(M, N, K, is_fp16);
    }
};

#endif // __HIP_PLATFORM_AMD__