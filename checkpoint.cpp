/**
 * checkpoint.cpp — Save/load checkpoint with merged multi-GPU state
 */

#include "common.h"

extern GPUContext g_gpu_contexts[MAX_GPUS];

typedef struct {
    uint32_t version;
    uint32_t dp_bits;
    uint64_t total_steps;
    uint64_t collision_count;
    uint64_t dp_insertions;
    uint64_t loop_restarts;
    uint32_t num_gpus;
    uint32_t reserved[2];
} CheckpointHeader;

void save_checkpoint(int dp_bits, int num_gpus) {
    printf("\nSaving checkpoint...\n");
    int dp_size = 1 << dp_bits;

    CompactDPEntry* h_merged = (CompactDPEntry*)calloc(dp_size, sizeof(CompactDPEntry));

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        CompactDPEntry* h_temp = (CompactDPEntry*)malloc(dp_size * sizeof(CompactDPEntry));
        cudaMemcpy(h_temp, g_gpu_contexts[g].d_table, dp_size * sizeof(CompactDPEntry), cudaMemcpyDeviceToHost);

        for (int i = 0; i < dp_size; i++) {
            if ((h_temp[i].metadata >> 31) & 1) {
                h_merged[i] = h_temp[i];
            }
        }
        free(h_temp);
    }

    uint64_t total_steps = 0, collision_count = 0, insertions = 0, loops = 0;
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        uint64_t ts, cc, ins, lr;
        cudaMemcpyFromSymbol(&ts, d_total_steps, sizeof(ts));
        cudaMemcpyFromSymbol(&cc, d_collision_count, sizeof(cc));
        cudaMemcpyFromSymbol(&ins, d_dp_insertions, sizeof(ins));
        cudaMemcpyFromSymbol(&lr, d_loop_restarts, sizeof(lr));
        total_steps += ts; collision_count += cc; insertions += ins; loops += lr;
    }

    FILE* f = fopen(CHECKPOINT_FILE, "wb");
    if (!f) { printf("  ERROR: Failed to open checkpoint file\n"); free(h_merged); return; }
    CheckpointHeader hdr = {CHECKPOINT_VERSION, (uint32_t)dp_bits, total_steps, collision_count, insertions, loops, (uint32_t)num_gpus, {0,0}};
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(h_merged, sizeof(CompactDPEntry), dp_size, f);
    fclose(f);
    printf("  Checkpoint saved: %s (%.1f MB)\n", CHECKPOINT_FILE, (sizeof(hdr) + dp_size * sizeof(CompactDPEntry)) / (1024.0 * 1024.0));
    free(h_merged);
}

int load_checkpoint(int* dp_bits, int num_gpus) {
    FILE* f = fopen(CHECKPOINT_FILE, "rb");
    if (!f) return 0;
    printf("Loading checkpoint...\n");
    CheckpointHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.version != CHECKPOINT_VERSION) {
        printf("  ERROR: Failed to read header or version mismatch\n");
        fclose(f);
        return 0;
    }
    *dp_bits = hdr.dp_bits;
    int dp_size = 1 << *dp_bits;

    CompactDPEntry* h_table = (CompactDPEntry*)malloc(dp_size * sizeof(CompactDPEntry));
    if (fread(h_table, sizeof(CompactDPEntry), dp_size, f) != (size_t)dp_size) {
        printf("  ERROR: Failed to read DP table\n");
        free(h_table);
        fclose(f);
        return 0;
    }
    fclose(f);

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMemcpy(g_gpu_contexts[g].d_table, h_table, dp_size * sizeof(CompactDPEntry), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_dp_table_size, &dp_size, sizeof(uint32_t));
        cudaMemcpyToSymbol(d_current_dp_bits, dp_bits, sizeof(*dp_bits));
        cudaMemcpyToSymbol(d_total_steps, &hdr.total_steps, sizeof(hdr.total_steps));
        cudaMemcpyToSymbol(d_collision_count, &hdr.collision_count, sizeof(hdr.collision_count));
        cudaMemcpyToSymbol(d_dp_insertions, &hdr.dp_insertions, sizeof(hdr.dp_insertions));
        cudaMemcpyToSymbol(d_loop_restarts, &hdr.loop_restarts, sizeof(hdr.loop_restarts));
    }
    free(h_table);
    printf("  Loaded: %lu steps, %lu collisions, %lu DPs\n",
           (unsigned long)hdr.total_steps, (unsigned long)hdr.collision_count, (unsigned long)hdr.dp_insertions);
    return 1;
}
