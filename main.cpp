/**
 * main.cpp — PSCKangaroo v57.0.0 entry point
 * CLI parsing, GPU setup, main search loop
 */

#include "common.h"
#include "kernel.cuh"

// Global state
int g_use_mod3_filter = 0;
int g_num_gpus = 1;
volatile int g_stop_signal = 0;

void sigint_handler(int sig) {
    printf("\nSIGINT received, saving checkpoint...\n");
    g_stop_signal = 1;
}

void hex_to_uint256(const char* hex, h_uint256_t* out) {
    memset(out, 0, sizeof(h_uint256_t));
    int len = strlen(hex);
    for (int i = 0; i < len && i < 64; i++) {
        int byte_val;
        char c = hex[len - 1 - i];
        if (c >= '0' && c <= '9') byte_val = c - '0';
        else if (c >= 'a' && c <= 'f') byte_val = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') byte_val = c - 'A' + 10;
        else byte_val = 0;
        int word = i >> 4;
        int nibble = (i & 15) << 2;
        out->w[word] |= ((uint64_t)byte_val << nibble);
    }
}

void print_usage(const char* prog) {
    printf("Usage: %s <pubkey_x> <pubkey_y> [options]\n", prog);
    printf("Options:\n");
    printf("  --benchmark     Run 10-second multi-stream benchmark\n");
    printf("  --mod3          Enable k ≡ 2 (mod 3) filter\n");
    printf("  --resume        Resume from checkpoint\n");
    printf("  --dp-bits N     Set DP table size (20-%d, default %d)\n", DP_BITS_MAX, DP_BITS_DEFAULT);
    printf("  --streams N     Number of CUDA streams per GPU (1-%d)\n", NUM_STREAMS_PER_GPU);
    printf("  --gpus N        Number of GPUs to use (1-%d, default 1)\n", MAX_GPUS);
    printf("  --help          Show this help\n");
    printf("\nFeatures v57.0.0:\n");
    printf("  - Montgomery domain arithmetic (O(1) reduction)\n");
    printf("  - GLV decomposition (k = k1 + k2*lambda)\n");
    printf("  - Pippenger multi-scalar multiplication\n");
    printf("  - Batch inversion (Montgomery trick)\n");
    printf("  - SoA layout + Cooperative Groups\n");
    printf("  - Multi-GPU with NVLink zero-copy sync\n");
    printf("  - Dynamic parallelism (adaptive grid)\n");
    printf("  - Tensor Cores support (sm_80+, compile with -DUSE_TENSOR_CORES)\n");
}

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("     PSCKangaroo v57.0.0 — Puzzle 135 Solver (REVOLUTIONARY)   \n");
    printf("     Montgomery | GLV | Pippenger | Tensor Cores | Multi-GPU    \n");
    printf("================================================================\n");
    printf("Puzzle %d: range [2^%d, 2^%d]\n", PUZZLE_NUMBER, RANGE_START_BITS, RANGE_END_BITS);
    printf("Multi-Herd: %d tame + %d wild = %d kangaroos/thread\n", NUM_TAME, NUM_WILD, KANGAROOS_PER_THREAD);
    printf("\n");

    signal(SIGINT, sigint_handler);
    int benchmark = 0, resume = 0, dp_bits = DP_BITS_DEFAULT, num_streams = NUM_STREAMS_PER_GPU;
    int num_gpus = 1;
    char* pubkey_x_hex = NULL, *pubkey_y_hex = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) benchmark = 1;
        else if (strcmp(argv[i], "--mod3") == 0) g_use_mod3_filter = 1;
        else if (strcmp(argv[i], "--resume") == 0) resume = 1;
        else if (strcmp(argv[i], "--dp-bits") == 0 && i+1 < argc) {
            dp_bits = atoi(argv[++i]);
            if (dp_bits < DP_BITS_MIN || dp_bits > DP_BITS_MAX) {
                printf("ERROR: dp-bits must be %d-%d\n", DP_BITS_MIN, DP_BITS_MAX);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--streams") == 0 && i+1 < argc) {
            num_streams = atoi(argv[++i]);
            if (num_streams < 1 || num_streams > NUM_STREAMS_PER_GPU) {
                printf("ERROR: streams must be 1-%d\n", NUM_STREAMS_PER_GPU);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--gpus") == 0 && i+1 < argc) {
            num_gpus = atoi(argv[++i]);
            if (num_gpus < 1 || num_gpus > MAX_GPUS) {
                printf("ERROR: gpus must be 1-%d\n", MAX_GPUS);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
        else if (!pubkey_x_hex) pubkey_x_hex = argv[i];
        else if (!pubkey_y_hex) pubkey_y_hex = argv[i];
    }

    // Validate GPU count
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) { printf("ERROR: No CUDA device found!\n"); return 1; }
    if (num_gpus > device_count) {
        printf("WARNING: Requested %d GPUs but only %d available. Using %d.\n", num_gpus, device_count, device_count);
        num_gpus = device_count;
    }
    g_num_gpus = num_gpus;

    if (g_use_mod3_filter) {
        printf("✓ k ≡ 2 (mod 3) filter ENABLED\n\n");
    } else {
        printf("✓ DEFAULT MODE: Montgomery, GLV, Pippenger, SOTA jumps\n\n");
    }

    printf("GPUs: %d\n", num_gpus);
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, g);
        printf("  GPU %d: %s | SMs: %d | Compute: %d.%d | Memory: %.2f GB\n",
               g, prop.name, prop.multiProcessorCount, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // Copy constants to all GPUs
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMemcpyToSymbol(d_p, SECP_P_64, sizeof(SECP_P_64));
        cudaMemcpyToSymbol(d_n, SECP_N_64, sizeof(SECP_N_64));
        cudaMemcpyToSymbol(d_Gx, SECP_GX_64, sizeof(SECP_GX_64));
        cudaMemcpyToSymbol(d_Gy, SECP_GY_64, sizeof(SECP_GY_64));
        cudaMemcpyToSymbol(d_p_half, SECP_P_HALF_64, sizeof(SECP_P_HALF_64));
        cudaMemcpyToSymbol(d_beta, GLV_BETA_64, sizeof(GLV_BETA_64));
        cudaMemcpyToSymbol(d_lambda, GLV_LAMBDA_64, sizeof(GLV_LAMBDA_64));
        cudaMemcpyToSymbol(d_two, TWO_64, sizeof(TWO_64));
    }

    if (benchmark) { 
        benchmark_v57(num_gpus); 
        return 0; 
    }

    h_AffinePoint target;
    if (pubkey_x_hex && pubkey_y_hex) {
        hex_to_uint256(pubkey_x_hex, (h_uint256_t*)&target.x);
        hex_to_uint256(pubkey_y_hex, (h_uint256_t*)&target.y);
    } else { printf("ERROR: Public key required\n"); print_usage(argv[0]); return 1; }

    printf("Target X: ");
    for (int i = NUM_WORDS_64 - 1; i >= 0; i--) printf("%016lx", target.x.w[i]);
    printf("\n");
    printf("Target Y: ");
    for (int i = NUM_WORDS_64 - 1; i >= 0; i--) printf("%016lx", target.y.w[i]);
    printf("\n\n");

    printf("Initializing...\n");
    precompute_jumps();
    precompute_t0();
    precompute_glv_tables();

    if (resume && load_checkpoint(&dp_bits, num_gpus)) printf("Resumed from checkpoint.\n");
    else init_dp_table_multi_gpu(dp_bits, num_gpus);

    // Convert target to Montgomery for device
    h_uint256_t target_x_mont, target_y_mont;
    h_to_mont(&target.x, &target_x_mont);
    h_to_mont(&target.y, &target_y_mont);

    Point h_target;
    memcpy(&h_target.x, &target_x_mont, sizeof(uint256_t));
    memcpy(&h_target.y, &target_y_mont, sizeof(uint256_t));
    h_target.z.w[0] = 1;
    for (int i = 1; i < NUM_WORDS_64; i++) h_target.z.w[i] = 0;

    cudaStream_t streams[MAX_GPUS][NUM_STREAMS_PER_GPU];
    dim3 grid[MAX_GPUS];
    dim3 block(THREADS_PER_BLOCK);

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, g);
        int num_blocks = prop.multiProcessorCount * BLOCKS_PER_SM;
        int max_threads = num_blocks * THREADS_PER_BLOCK;
        if (max_threads > MAX_TOTAL_THREADS) {
            num_blocks = MAX_TOTAL_THREADS / THREADS_PER_BLOCK;
            printf("WARNING GPU %d: Limiting to %d blocks for thread safety\n", g, num_blocks);
        }
        grid[g] = dim3(num_blocks);

        for (int s = 0; s < num_streams; s++) {
            cudaStreamCreate(&streams[g][s]);
        }
    }

    uint32_t dp_table_size = 1 << dp_bits;
    int use_mod3 = g_use_mod3_filter ? 1 : 0;

    printf("\nLaunch: %d blocks x %d threads x %d streams x %d GPUs\n", 
           grid[0].x, THREADS_PER_BLOCK, num_streams, num_gpus);
    printf("DP bits: %d (24-byte aligned, 192-bit dist)\n", dp_bits);
    printf("Steps per kernel: %llu (%.2fM)\n", (unsigned long long)STEPS_PER_KERNEL,
           STEPS_PER_KERNEL / 1000000.0);
    printf("Features: Montgomery O(1), GLV, Pippenger, BatchInv, SoA, CG\n");
    printf("Multi-GPU sync: pinned memory (NVLink auto-detect)\n");
    printf("\n");

    uint64_t seed = (uint64_t)time(NULL);
    printf("Starting search (seed=%llu). Press Ctrl+C to save checkpoint.\n\n", (unsigned long long)seed);

    clock_t start = clock();
    time_t last_checkpoint = time(NULL), last_status = time(NULL), last_dp_adjust = time(NULL), last_ipc_sync = time(NULL);

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        for (int s = 0; s < num_streams; s++) {
            kangaroo_kernel_v57<<<grid[g], block, 0, streams[g][s]>>>(h_target, seed + g * num_streams + s, dp_bits, dp_table_size, use_mod3);
        }
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("KERNEL LAUNCH ERROR: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }

    uint32_t found = 0;
    uint64_t last_steps = 0;

    while (!found && !g_stop_signal) {
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }

        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            uint32_t gpu_found;
            cudaMemcpyFromSymbol(&gpu_found, d_found, sizeof(gpu_found));
            if (gpu_found) { found = 1; break; }
        }

        uint64_t total_steps = 0, collisions = 0, insertions = 0, loops = 0;
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            uint64_t ts, cc, ins, lr;
            cudaMemcpyFromSymbol(&ts, d_total_steps, sizeof(ts));
            cudaMemcpyFromSymbol(&cc, d_collision_count, sizeof(cc));
            cudaMemcpyFromSymbol(&ins, d_dp_insertions, sizeof(ins));
            cudaMemcpyFromSymbol(&lr, d_loop_restarts, sizeof(lr));
            total_steps += ts; collisions += cc; insertions += ins; loops += lr;
        }

        time_t now = time(NULL);

        if (now - last_ipc_sync >= IPC_SYNC_INTERVAL_SEC) {
            sync_dp_table_between_gpus(num_gpus);
            last_ipc_sync = now;
        }

        if (now - last_dp_adjust >= 60) {
            int new_dp_bits = calculate_adaptive_dp_bits(total_steps, insertions, dp_bits);
            if (new_dp_bits != dp_bits) {
                printf("  Adjusting DP bits: %d -> %d\n", dp_bits, new_dp_bits);
                dp_bits = new_dp_bits;
                dp_table_size = 1 << dp_bits;
                for (int g = 0; g < num_gpus; g++) {
                    cudaSetDevice(g);
                    cudaMemcpyToSymbol(d_current_dp_bits, &dp_bits, sizeof(dp_bits));
                    cudaMemcpyToSymbol(d_dp_table_size, &dp_table_size, sizeof(uint32_t));
                }
                last_dp_adjust = now;
            }
        }

        if (now - last_status >= 5 && total_steps > last_steps) {
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            double rate = total_steps / elapsed;

            double range_width = pow(2.0, PUZZLE_NUMBER);
            int total_kangaroos = 0;
            for (int g = 0; g < num_gpus; g++) {
                total_kangaroos += grid[g].x * THREADS_PER_BLOCK * KANGAROOS_PER_THREAD * num_streams;
            }
            double effective_range = g_use_mod3_filter ? range_width / 3.0 : range_width;
            double expected_steps = 2.0 * sqrt(effective_range / (double)total_kangaroos);
            double progress = 100.0 * (double)total_steps / expected_steps;
            if (progress > 100.0) progress = 100.0;

            printf("  Steps: %12lu | DPs: %10lu | Loops: %5lu | Rate: %.3e | Time: %.1fs | Progress: %.6f%%\n",
                   (unsigned long)total_steps, (unsigned long)insertions,
                   (unsigned long)loops, rate, elapsed, progress);
            last_steps = total_steps;
            last_status = now;
        }

        if (now - last_checkpoint >= CHECKPOINT_INTERVAL_SEC) {
            save_checkpoint(dp_bits, num_gpus);
            last_checkpoint = now;
        }
        if (found) break;

        seed += num_gpus * num_streams;
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            for (int s = 0; s < num_streams; s++) {
                kangaroo_kernel_v57<<<grid[g], block, 0, streams[g][s]>>>(h_target, seed + g * num_streams + s, dp_bits, dp_table_size, use_mod3);
            }
        }
    }

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        for (int s = 0; s < num_streams; s++) {
            cudaStreamDestroy(streams[g][s]);
        }
    }

    if (g_stop_signal) { printf("\nInterrupted by user.\n"); save_checkpoint(dp_bits, num_gpus); return 0; }

    int winner_gpu = 0;
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        uint32_t gpu_found;
        cudaMemcpyFromSymbol(&gpu_found, d_found, sizeof(gpu_found));
        if (gpu_found) { winner_gpu = g; break; }
    }

    cudaSetDevice(winner_gpu);
    h_uint256_t result_k;
    cudaMemcpyFromSymbol(&result_k, d_result_k, sizeof(result_k));
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("\n================================================================\n");
    printf("                    K E Y   F O U N N D !                       \n");
    printf("                    GPU %d reported collision                   \n", winner_gpu);
    printf("================================================================\n");
    printf("  diff = ");
    for (int i = NUM_WORDS_64 - 1; i >= 0; i--) printf("%016lx", result_k.w[i]);
    printf("\n");
    printf("  Time: %.2f seconds (%.4f hours)\n", elapsed, elapsed / 3600.0);
    printf("================================================================\n");

    if (verify_key(&result_k, &target)) {
        save_checkpoint(dp_bits, num_gpus);
        return 0;
    }

    return 1;
}
