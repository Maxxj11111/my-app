/**
 * benchmark.cpp — Multi-GPU benchmark with adaptive parameters
 */

#include "common.h"
#include "kernel.cuh"

void benchmark_v57(int num_gpus) {
    printf("\n=== BENCHMARK v57.0.0 ===\n");
    printf("GPUs: %d\n", num_gpus);
    printf("Features: Montgomery, SoA, CG, Multi-GPU, Prefetch, Branchless\n");
    if (g_use_mod3_filter) printf("Mod3 filter: ON\n");

    init_dp_table_multi_gpu(DP_BITS_DEFAULT, num_gpus);
    precompute_jumps();
    precompute_t0();
    precompute_glv_tables();

    // Test target: 2^100 * G
    h_AffinePoint G, test_target;
    memcpy(G.x.w, SECP_GX_64, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(G.y.w, SECP_GY_64, sizeof(uint64_t) * NUM_WORDS_64);
    h_to_mont((const h_uint256_t*)SECP_GX_64, &G.x);
    h_to_mont((const h_uint256_t*)SECP_GY_64, &G.y);

    h_uint256_t test_scalar;
    memset(&test_scalar, 0, sizeof(test_scalar));
    test_scalar.w[1] = 1ULL << 36;
    h_scalar_mult_windowed(&test_scalar, &G, &test_target);
    printf("Benchmark target: 2^100 * G\n");

    Point h_target;
    memcpy(&h_target.x, &test_target.x, sizeof(uint256_t));
    memcpy(&h_target.y, &test_target.y, sizeof(uint256_t));
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
        }
        grid[g] = dim3(num_blocks);

        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            cudaStreamCreate(&streams[g][s]);
        }
    }

    clock_t start = clock();
    uint64_t seed = (uint64_t)time(NULL);
    uint32_t dp_table_size = 1 << DP_BITS_DEFAULT;
    int use_mod3 = g_use_mod3_filter ? 1 : 0;

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            kangaroo_kernel_v57<<<grid[g], block, 0, streams[g][s]>>>(h_target, seed + g * NUM_STREAMS_PER_GPU + s, DP_BITS_DEFAULT, dp_table_size, use_mod3);
        }
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("KERNEL LAUNCH ERROR: %s\n", cudaGetErrorString(launch_err));
        return;
    }

    printf("Running 10-second multi-GPU benchmark...\n");
    printf("  Steps        | DPs        | Loops      | Rate        | Time\n");
    printf("  -------------|------------|------------|-------------|-------\n");

    double last_print = 0.0;
    while ((double)(clock() - start) / CLOCKS_PER_SEC < 10.0) {
        bool all_done = true;
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaError_t err = cudaStreamQuery(streams[g][0]);
            if (err != cudaSuccess) all_done = false;
        }

        if (all_done) {
            for (int g = 0; g < num_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
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

            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            double rate = total_steps / elapsed;
            printf("  %12lu | %10lu | %10lu | %.3e | %.1fs\n",
                   (unsigned long)total_steps, (unsigned long)insertions,
                   (unsigned long)loops, rate, elapsed);

            seed += num_gpus * NUM_STREAMS_PER_GPU;
            for (int g = 0; g < num_gpus; g++) {
                cudaSetDevice(g);
                for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
                    kangaroo_kernel_v57<<<grid[g], block, 0, streams[g][s]>>>(h_target, seed + g * NUM_STREAMS_PER_GPU + s, DP_BITS_DEFAULT, dp_table_size, use_mod3);
                }
            }
            launch_err = cudaGetLastError();
            if (launch_err != cudaSuccess) {
                printf("KERNEL RE-LAUNCH ERROR: %s\n", cudaGetErrorString(launch_err));
                break;
            }
        }

        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        if (elapsed - last_print >= 2.0) {
            for (int g = 0; g < num_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
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

            double rate = total_steps / elapsed;
            printf("  %12lu | %10lu | %10lu | %.3e | %.1fs\n",
                   (unsigned long)total_steps, (unsigned long)insertions,
                   (unsigned long)loops, rate, elapsed);
            last_print = elapsed;
        }
        usleep(100000);
    }

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            cudaStreamDestroy(streams[g][s]);
        }
    }
}
