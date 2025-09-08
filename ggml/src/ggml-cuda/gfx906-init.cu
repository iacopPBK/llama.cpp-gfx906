#include "common.cuh"
#include "gfx906-config.cuh"

#include <cstdio>
#include <cstring>

#ifdef GGML_HIP_GFX906_OPTIMIZED

// GFX906 Device Information Structure
struct gfx906_device_info {
    int    device_id;
    char   name[256];
    int    compute_units;
    size_t lds_size;
    int    wave_size;
    size_t global_mem_size;
    int    max_threads_per_block;
    bool   initialized;
};

static gfx906_device_info g_gfx906_devices[GGML_CUDA_MAX_DEVICES];
static int                g_gfx906_device_count = 0;
static bool               g_gfx906_initialized  = false;

// Initialize GFX906 device detection and configuration
extern "C" bool ggml_cuda_gfx906_init() {
    if (g_gfx906_initialized) {
        return true;
    }

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    g_gfx906_device_count = 0;

    for (int i = 0; i < device_count && i < GGML_CUDA_MAX_DEVICES; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        // Check if this is a GFX906 device
        if (strstr(prop.gcnArchName, "gfx906") != nullptr) {
            gfx906_device_info & info = g_gfx906_devices[g_gfx906_device_count];

            info.device_id = i;
            strncpy(info.name, prop.name, sizeof(info.name) - 1);
            info.name[sizeof(info.name) - 1] = '\0';
            info.compute_units               = prop.multiProcessorCount;
            info.lds_size                    = GFX906_LDS_SIZE;
            info.wave_size                   = GFX906_WAVE_SIZE;
            info.global_mem_size             = prop.totalGlobalMem;
            info.max_threads_per_block       = prop.maxThreadsPerBlock;
            info.initialized                 = true;

            // Log device information
            GGML_LOG_INFO("GFX906 Device %d: %s\n", i, info.name);
            GGML_LOG_INFO("  Compute Units: %d\n", info.compute_units);
            GGML_LOG_INFO("  LDS Size: %zu KB\n", info.lds_size / 1024);
            GGML_LOG_INFO("  Wave Size: %d\n", info.wave_size);
            GGML_LOG_INFO("  Global Memory: %.2f GB\n", info.global_mem_size / (1024.0 * 1024.0 * 1024.0));
            GGML_LOG_INFO("  Max Threads/Block: %d\n", info.max_threads_per_block);

            // Verify expected configuration
            if (info.compute_units != GFX906_NUM_CUS) {
                GGML_LOG_WARN("  Warning: Expected %d CUs, found %d\n", GFX906_NUM_CUS, info.compute_units);
            }

            g_gfx906_device_count++;
        }
    }

    if (g_gfx906_device_count > 0) {
        GGML_LOG_INFO("Found %d GFX906 device(s)\n", g_gfx906_device_count);
        g_gfx906_initialized = true;
        return true;
    } else {
        GGML_LOG_INFO("No GFX906 devices found\n");
        return false;
    }
}

// Get GFX906 device info by index
extern "C" const gfx906_device_info * ggml_cuda_gfx906_get_device_info(int device_id) {
    if (!g_gfx906_initialized) {
        ggml_cuda_gfx906_init();
    }

    for (int i = 0; i < g_gfx906_device_count; i++) {
        if (g_gfx906_devices[i].device_id == device_id) {
            return &g_gfx906_devices[i];
        }
    }

    return nullptr;
}

// Stream management for GFX906
struct gfx906_stream_pool {
    cudaStream_t streams[GFX906_MAX_STREAMS];
    bool         in_use[GFX906_MAX_STREAMS];
    int          device_id;
    int          num_streams;
};

static gfx906_stream_pool g_stream_pools[GGML_CUDA_MAX_DEVICES];

// Initialize stream pool for a GFX906 device
extern "C" bool ggml_cuda_gfx906_init_streams(int device_id) {
    const gfx906_device_info * info = ggml_cuda_gfx906_get_device_info(device_id);
    if (!info) {
        return false;
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    gfx906_stream_pool & pool = g_stream_pools[device_id];
    pool.device_id            = device_id;
    pool.num_streams          = GFX906_DEFAULT_STREAMS;

    for (int i = 0; i < pool.num_streams; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&pool.streams[i], cudaStreamNonBlocking));
        pool.in_use[i] = false;
    }

    GGML_LOG_INFO("Initialized %d streams for GFX906 device %d\n", pool.num_streams, device_id);
    return true;
}

// Get an available stream from the pool
extern "C" cudaStream_t ggml_cuda_gfx906_get_stream(int device_id) {
    gfx906_stream_pool & pool = g_stream_pools[device_id];

    for (int i = 0; i < pool.num_streams; i++) {
        if (!pool.in_use[i]) {
            pool.in_use[i] = true;
            return pool.streams[i];
        }
    }

    // If no stream available, return default stream
    return 0;
}

// Release a stream back to the pool
extern "C" void ggml_cuda_gfx906_release_stream(int device_id, cudaStream_t stream) {
    if (stream == 0) {
        return;  // Default stream
    }

    gfx906_stream_pool & pool = g_stream_pools[device_id];

    for (int i = 0; i < pool.num_streams; i++) {
        if (pool.streams[i] == stream) {
            pool.in_use[i] = false;
            return;
        }
    }
}

// Cleanup streams for a device
extern "C" void ggml_cuda_gfx906_cleanup_streams(int device_id) {
    gfx906_stream_pool & pool = g_stream_pools[device_id];

    CUDA_CHECK(cudaSetDevice(device_id));

    for (int i = 0; i < pool.num_streams; i++) {
        if (pool.streams[i]) {
            CUDA_CHECK(cudaStreamDestroy(pool.streams[i]));
            pool.streams[i] = nullptr;
        }
        pool.in_use[i] = false;
    }

    pool.num_streams = 0;
}

// Performance monitoring
struct gfx906_perf_counter {
    const char * name;
    uint64_t     count;
    double       total_time_ms;
    double       min_time_ms;
    double       max_time_ms;
};

static gfx906_perf_counter g_perf_counters[64];
static int                 g_num_perf_counters = 0;

// Register a performance counter
extern "C" int ggml_cuda_gfx906_register_perf_counter(const char * name) {
    if (g_num_perf_counters >= 64) {
        return -1;
    }

    int                   id      = g_num_perf_counters++;
    gfx906_perf_counter & counter = g_perf_counters[id];
    counter.name                  = name;
    counter.count                 = 0;
    counter.total_time_ms         = 0.0;
    counter.min_time_ms           = 1e9;
    counter.max_time_ms           = 0.0;

    return id;
}

// Update performance counter
extern "C" void ggml_cuda_gfx906_update_perf_counter(int counter_id, double time_ms) {
    if (counter_id < 0 || counter_id >= g_num_perf_counters) {
        return;
    }

    gfx906_perf_counter & counter = g_perf_counters[counter_id];
    counter.count++;
    counter.total_time_ms += time_ms;
    counter.min_time_ms = (time_ms < counter.min_time_ms) ? time_ms : counter.min_time_ms;
    counter.max_time_ms = (time_ms > counter.max_time_ms) ? time_ms : counter.max_time_ms;
}

// Print performance statistics
extern "C" void ggml_cuda_gfx906_print_perf_stats() {
    GGML_LOG_INFO("GFX906 Performance Statistics:\n");
    GGML_LOG_INFO(
        "%-30s %10s %12s %12s %12s %12s\n", "Operation", "Count", "Total (ms)", "Avg (ms)", "Min (ms)", "Max (ms)");
    GGML_LOG_INFO(
        "-"
        "%.0s",
        "----------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < g_num_perf_counters; i++) {
        const gfx906_perf_counter & counter = g_perf_counters[i];
        if (counter.count > 0) {
            double avg_time = counter.total_time_ms / counter.count;
            GGML_LOG_INFO("%-30s %10llu %12.3f %12.3f %12.3f %12.3f\n",
                          counter.name,
                          (unsigned long long) counter.count,
                          counter.total_time_ms,
                          avg_time,
                          counter.min_time_ms,
                          counter.max_time_ms);
        }
    }
}

// Memory pool management for GFX906
struct gfx906_memory_pool {
    void * base_ptr;
    size_t total_size;
    size_t used_size;
    int    device_id;
};

static gfx906_memory_pool g_memory_pools[GGML_CUDA_MAX_DEVICES];

// Initialize memory pool for GFX906 device
extern "C" bool ggml_cuda_gfx906_init_memory_pool(int device_id, size_t pool_size) {
    const gfx906_device_info * info = ggml_cuda_gfx906_get_device_info(device_id);
    if (!info) {
        return false;
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    gfx906_memory_pool & pool = g_memory_pools[device_id];
    pool.device_id            = device_id;
    pool.total_size           = gfx906_align_memory(pool_size);
    pool.used_size            = 0;

    CUDA_CHECK(cudaMalloc(&pool.base_ptr, pool.total_size));

    GGML_LOG_INFO("Initialized memory pool of %.2f GB for GFX906 device %d\n",
                  pool.total_size / (1024.0 * 1024.0 * 1024.0),
                  device_id);

    return true;
}

// Allocate from memory pool
extern "C" void * ggml_cuda_gfx906_pool_alloc(int device_id, size_t size) {
    gfx906_memory_pool & pool = g_memory_pools[device_id];

    size_t aligned_size = gfx906_align_memory(size);

    if (pool.used_size + aligned_size > pool.total_size) {
        GGML_LOG_ERROR("GFX906 memory pool exhausted on device %d\n", device_id);
        return nullptr;
    }

    void * ptr = (char *) pool.base_ptr + pool.used_size;
    pool.used_size += aligned_size;

    return ptr;
}

// Reset memory pool
extern "C" void ggml_cuda_gfx906_pool_reset(int device_id) {
    gfx906_memory_pool & pool = g_memory_pools[device_id];
    pool.used_size            = 0;
}

// Cleanup memory pool
extern "C" void ggml_cuda_gfx906_cleanup_memory_pool(int device_id) {
    gfx906_memory_pool & pool = g_memory_pools[device_id];

    if (pool.base_ptr) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaFree(pool.base_ptr));
        pool.base_ptr   = nullptr;
        pool.total_size = 0;
        pool.used_size  = 0;
    }
}

// Global cleanup for all GFX906 resources
extern "C" void ggml_cuda_gfx906_cleanup() {
    for (int i = 0; i < g_gfx906_device_count; i++) {
        int device_id = g_gfx906_devices[i].device_id;
        ggml_cuda_gfx906_cleanup_streams(device_id);
        ggml_cuda_gfx906_cleanup_memory_pool(device_id);
    }

    g_gfx906_initialized  = false;
    g_gfx906_device_count = 0;

    GGML_LOG_INFO("GFX906 backend cleaned up\n");
}

#endif  // GGML_HIP_GFX906_OPTIMIZED

