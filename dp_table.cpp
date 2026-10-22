/**
 * dp_table.cpp — Multi-GPU DP table with async sync
 * - Pinned memory staging
 * - Zero-copy via NVLink (if available)
 * - Adaptive DP bits
 */

#include "common.h"

typedef struct {
    CompactDPEntry* d_table;
    uint32_t dp_bits;
    uint32_t dp_size;
    int device_id;
} GPUContext;

static GPUContext g_gpu_contexts[MAX_GPUS];
static int g_num_active_gpus = 1;

// Pinned host buffer for async transfers
static CompactDPEntry* g_h_pinned = NULL;
static int g_pinned_size = 0;

void init_dp_table_multi_gpu(int dp_bits, int num_gpus) {
    printf("Initializing DP table (%d bits = %d entries) across %d GPU(s)...\n", dp_bits, 1 << dp_bits, num_gpus);
    int dp_size = 1 << dp_bits;

    CompactDPEntry* h_table = (CompactDPEntry*)calloc(dp_size, sizeof(CompactDPEntry));
    if (!h_table) { printf("ERROR: Failed to allocate DP table\n"); exit(1); }

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);

        CompactDPEntry* d_table;
        cudaError_t err = cudaMalloc(&d_table, dp_size * sizeof(CompactDPEntry));
        if (err != cudaSuccess) {
            printf("ERROR: Failed to allocate DP table on GPU %d: %s\n", g, cudaGetErrorString(err));
            exit(1);
        }

        cudaMemcpy(d_table, h_table, dp_size * sizeof(CompactDPEntry), cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(d_dp_table, &d_table, sizeof(CompactDPEntry*));
        cudaMemcpyToSymbol(d_dp_table_size, &dp_size, sizeof(uint32_t));

        g_gpu_contexts[g].d_table = d_table;
        g_gpu_contexts[g].dp_bits = dp_bits;
        g_gpu_contexts[g].dp_size = dp_size;
        g_gpu_contexts[g].device_id = g;
    }
    free(h_table);

    // Allocate pinned memory for async transfers
    if (g_h_pinned) cudaFreeHost(g_h_pinned);
    cudaMallocHost(&g_h_pinned, dp_size * sizeof(CompactDPEntry));
    g_pinned_size = dp_size;

    uint32_t zero = 0;
    uint64_t zero64 = 0;
    h_uint256_t zero256;
    memset(&zero256, 0, sizeof(zero256));

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMemcpyToSymbol(d_found, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_current_dp_bits, &dp_bits, sizeof(dp_bits));
        cudaMemcpyToSymbol(d_result_k, &zero256, sizeof(uint256_t));
        cudaMemcpyToSymbol(d_total_steps, &zero64, sizeof(zero64));
        cudaMemcpyToSymbol(d_collision_count, &zero64, sizeof(zero64));
        cudaMemcpyToSymbol(d_dp_insertions, &zero64, sizeof(zero64));
        cudaMemcpyToSymbol(d_loop_restarts, &zero64, sizeof(zero64));
    }

    printf("  DP table: %d entries x 24 bytes = %.2f MB per GPU\n", dp_size, dp_size * 24.0 / (1024.0 * 1024.0));
    printf("  Total DP memory: %.2f MB\n", num_gpus * dp_size * 24.0 / (1024.0 * 1024.0));
}

// Async DP table sync between GPUs
void sync_dp_table_between_gpus(int num_gpus) {
    if (num_gpus <= 1) return;

    int dp_size = g_gpu_contexts[0].dp_size;

    // Stage from GPU 0 to pinned host memory
    cudaSetDevice(0);
    cudaStream_t sync_stream;
    cudaStreamCreate(&sync_stream);
    cudaMemcpyAsync(g_h_pinned, g_gpu_contexts[0].d_table, dp_size * sizeof(CompactDPEntry), 
                    cudaMemcpyDeviceToHost, sync_stream);
    cudaStreamSynchronize(sync_stream);
    cudaStreamDestroy(sync_stream);

    // Broadcast to all other GPUs
    for (int g = 1; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(g_gpu_contexts[g].d_table, g_h_pinned, dp_size * sizeof(CompactDPEntry),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamDestroy(stream);
    }
}

// Zero-copy sync via NVLink (if peer access enabled)
void sync_dp_table_nvlink(int num_gpus) {
    if (num_gpus <= 1) return;

    // Check if peer access is available
    int can_access;
    cudaDeviceCanAccessPeer(&can_access, 1, 0);
    if (!can_access) {
        sync_dp_table_between_gpus(num_gpus);
        return;
    }

    // Enable peer access
    for (int g = 1; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceEnablePeerAccess(0, 0);
    }

    // Direct GPU-to-GPU copy
    int dp_size = g_gpu_contexts[0].dp_size;
    for (int g = 1; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMemcpyPeer(g_gpu_contexts[g].d_table, g, 
                       g_gpu_contexts[0].d_table, 0,
                       dp_size * sizeof(CompactDPEntry));
    }

    // Disable peer access
    for (int g = 1; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceDisablePeerAccess(0);
    }
}

int calculate_adaptive_dp_bits(uint64_t total_steps, uint64_t dp_insertions, int current_dp_bits) {
    if (total_steps < 1000000) return current_dp_bits;
    double dp_rate = (double)dp_insertions / total_steps;
    double target_rate = 1.0 / (1ULL << current_dp_bits);
    if (dp_rate > target_rate * 1.5 && current_dp_bits < DP_BITS_MAX) return current_dp_bits + 1;
    if (dp_rate < target_rate * 0.5 && current_dp_bits > DP_BITS_MIN) return current_dp_bits - 1;
    return current_dp_bits;
}
