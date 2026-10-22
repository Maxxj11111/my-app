/**
 * kernel.cu — Main GPU kernel v57.0.0
 * - Montgomery domain EC operations
 * - SoA layout
 * - Cooperative Groups
 * - Dynamic parallelism support
 * - Multi-herd: 2 tame + 4 wild
 */

#include "common.h"
#include "mont_math.cuh"
#include "ec_ops.cuh"
#include "hash_dp.cuh"

// ============================================================================
// DEVICE CONSTANT DEFINITIONS
// ============================================================================

__constant__ uint64_t d_p[NUM_WORDS_64];
__constant__ uint64_t d_n[NUM_WORDS_64];
__constant__ uint64_t d_Gx[NUM_WORDS_64];
__constant__ uint64_t d_Gy[NUM_WORDS_64];
__constant__ uint64_t d_T0x[NUM_WORDS_64];
__constant__ uint64_t d_T0y[NUM_WORDS_64];
__constant__ uint64_t d_p_half[NUM_WORDS_64];
__constant__ uint64_t d_beta[NUM_WORDS_64];
__constant__ uint64_t d_lambda[NUM_WORDS_64];
__constant__ uint64_t d_two[NUM_WORDS_64];
__constant__ uint64_t d_mont_r2[NUM_WORDS_64] = {{0x0000000000000000ULL, 0x0000000000000000ULL, 
                                                     0x0000000000000000ULL, 0x0000000000000001ULL}};

__constant__ AffinePoint c_jump_points[NUM_JUMPS];
__constant__ uint64_t c_jump_dist[NUM_JUMPS];

__constant__ AffinePoint c_glv_G1;
__constant__ AffinePoint c_glv_G2;

__device__ CompactDPEntry* d_dp_table;
__device__ uint256_t d_result_k;
__device__ uint32_t d_found;
__device__ uint64_t d_total_steps;
__device__ uint64_t d_collision_count;
__device__ uint64_t d_dp_insertions;
__device__ uint64_t d_loop_restarts;
__device__ uint32_t d_current_dp_bits;
__device__ uint32_t d_dp_table_size;

// ============================================================================
// MODULAR ARITHMETIC (mod n) — for distance tracking
// ============================================================================

__device__ __forceinline__ void add256_mod_n(const uint256_t* a, const uint256_t* b, uint256_t* r) {
    uint32_t carry = add256_v2(a, b, r);
    uint256_t t;
    int need_reduce = carry || cmp256(r, (const uint256_t*)d_n) >= 0;
    sub256(r, (const uint256_t*)d_n, &t);
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        r->w[i] = need_reduce ? t.w[i] : r->w[i];
    }
}

__device__ __forceinline__ void sub256_mod_n(const uint256_t* a, const uint256_t* b, uint256_t* r) {
    uint32_t borrow = sub256(a, b, r);
    uint256_t t;
    add256_v2(r, (const uint256_t*)d_n, &t);
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        r->w[i] = borrow ? t.w[i] : r->w[i];
    }
}

// ============================================================================
// MOD3 FILTER
// ============================================================================

__device__ __forceinline__ uint32_t mod3_u64(uint64_t x) {
    x = (x >> 32) + (x & 0xFFFFFFFFULL);
    x = (x >> 16) + (x & 0xFFFF);
    x = (x >> 8) + (x & 0xFF);
    x = (x >> 4) + (x & 0xF);
    x = (x >> 2) + (x & 0x3);
    x = (x >> 2) + (x & 0x3);
    x -= (x >= 3) ? 3 : 0;
    x -= (x >= 3) ? 3 : 0;
    return (uint32_t)x;
}

__device__ __forceinline__ uint32_t mod3_256(const uint256_t* a) {
    uint32_t sum = mod3_u64(a->w[0]) + mod3_u64(a->w[1]) 
                 + mod3_u64(a->w[2]) + mod3_u64(a->w[3]);
    sum = (sum >> 2) + (sum & 3);
    sum -= (sum >= 3) ? 3 : 0;
    return sum;
}

__device__ __forceinline__ void adjust_mod3(uint256_t* dist, uint32_t target_mod3) {
    uint32_t r = mod3_256(dist);
    uint32_t delta = (target_mod3 + 3 - r);
    delta -= (delta >= 3) ? 3 : 0;
    uint64_t mask = -(int64_t)(delta != 0);
    uint256_t adj = {{0}};
    adj.w[0] = delta & mask;
    add256_mod_n(dist, &adj, dist);
}

// ============================================================================
// RNG
// ============================================================================

__device__ __forceinline__ uint64_t xorshift64star(uint64_t* state) {
    uint64_t s = *state;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    *state = s;
    return s * 0x2545F4914F6CDD1DULL;
}

__device__ __forceinline__ void rand135_range(uint64_t* rng, uint256_t* out) {
    uint64_t s = *rng;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    out->w[0] = s * 0x2545F4914F6CDD1DULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    out->w[1] = s * 0x2545F4914F6CDD1DULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    out->w[2] = (s * 0x2545F4914F6CDD1DULL & 0x3F) | 0x40;
    out->w[3] = 0;
    *rng = s;
}

__device__ __forceinline__ void mul256_by_u64_mod_n(const uint256_t* a, uint64_t m, uint256_t* r) {
    unsigned __int128 prod0 = (unsigned __int128)a->w[0] * m;
    unsigned __int128 prod1 = (unsigned __int128)a->w[1] * m + (uint64_t)(prod0 >> 64);
    unsigned __int128 prod2 = (unsigned __int128)a->w[2] * m + (uint64_t)(prod1 >> 64);
    unsigned __int128 prod3 = (unsigned __int128)a->w[3] * m + (uint64_t)(prod2 >> 64);

    uint256_t temp = {{(uint64_t)prod0, (uint64_t)prod1, (uint64_t)prod2, (uint64_t)prod3}};

    while (cmp256(&temp, (const uint256_t*)d_n) >= 0) {
        uint256_t t;
        sub256(&temp, (const uint256_t*)d_n, &t);
        temp = t;
    }
    *r = temp;
}

// ============================================================================
// MAIN KERNEL v57.0.0
// ============================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
kangaroo_kernel_v57(Point target, uint64_t seed, uint32_t dp_bits, uint32_t dp_table_size, int use_mod3) {
    int tid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    int lane = threadIdx.x & 31;

    if (tid >= MAX_TOTAL_THREADS) return;
    if (d_dp_table == NULL) return;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Shared memory for jump points
    __shared__ AffinePoint s_jump_points[NUM_JUMPS];
    __shared__ uint64_t s_jump_dist[NUM_JUMPS];
    int load_idx = threadIdx.x;
    #pragma unroll 4
    for (int i = load_idx; i < NUM_JUMPS; i += THREADS_PER_BLOCK) {
        s_jump_points[i] = c_jump_points[i];
        s_jump_dist[i] = c_jump_dist[i];
    }
    __syncthreads();

    // SoA state
    KangarooSoA soa;

    // Thread offset = tid * spacing
    uint256_t spacing = {{0, 0, 1ULL << 28, 0}};  // 2^92
    uint256_t thread_offset;
    mul256_by_u64_mod_n(&spacing, (uint64_t)tid, &thread_offset);

    uint256_t herd_spacing = {{0, 1ULL << 3, 0, 0}};  // 2^67

    if (use_mod3) {
        adjust_mod3(&thread_offset, 0);
        adjust_mod3(&herd_spacing, 0);
    }

    // Initialize TAME (with negation map)
    uint256_t T0x, T0y;
    T0x.w[0] = d_T0x[0]; T0x.w[1] = d_T0x[1]; T0x.w[2] = d_T0x[2]; T0x.w[3] = d_T0x[3];
    T0y.w[0] = d_T0y[0]; T0y.w[1] = d_T0y[1]; T0y.w[2] = d_T0y[2]; T0y.w[3] = d_T0y[3];

    #pragma unroll
    for (int k = 0; k < NUM_TAME; k++) {
        soa_to_jacobian(&soa, k, &T0x, &T0y);
        uint64_t* dp = soa_dist_ptr(&soa, k);
        dp[0] = thread_offset.w[0]; dp[1] = thread_offset.w[1];
        dp[2] = thread_offset.w[2]; dp[3] = thread_offset.w[3];
        if (k == 1) {
            uint256_t dist;
            soa_load_dist(&soa, k, &dist);
            add256_mod_n(&dist, &herd_spacing, &dist);
            soa_store_dist(&soa, k, &dist);
        }
        if (use_mod3) {
            uint256_t dist;
            soa_load_dist(&soa, k, &dist);
            adjust_mod3(&dist, 0);
            soa_store_dist(&soa, k, &dist);
        }
        soa.is_tame[k] = 1;
        soa.last_hash[k] = 0;
        soa.loop_counter[k] = 0;
    }

    // Initialize WILD (NO negation map)
    uint64_t rng = ((uint64_t)tid << 32) ^ seed;
    #pragma unroll
    for (int k = NUM_TAME; k < KANGAROOS_PER_THREAD; k++) {
        uint64_t* xp = soa_x_ptr(&soa, k);
        uint64_t* yp = soa_y_ptr(&soa, k);
        uint64_t* zp = soa_z_ptr(&soa, k);
        xp[0] = target.x.w[0]; xp[1] = target.x.w[1]; xp[2] = target.x.w[2]; xp[3] = target.x.w[3];
        yp[0] = target.y.w[0]; yp[1] = target.y.w[1]; yp[2] = target.y.w[2]; yp[3] = target.y.w[3];
        zp[0] = 1; zp[1] = 0; zp[2] = 0; zp[3] = 0;

        uint256_t dist;
        rand135_range(&rng, &dist);
        soa_store_dist(&soa, k, &dist);
        if (use_mod3) {
            uint256_t d;
            soa_load_dist(&soa, k, &d);
            adjust_mod3(&d, 2);
            soa_store_dist(&soa, k, &d);
        }
        soa.is_tame[k] = 0;
        soa.last_hash[k] = 0;
        soa.loop_counter[k] = 0;
    }

    uint32_t dp_mask = (1U << dp_bits) - 1;
    uint64_t warp_batch_steps = 0;

    for (uint64_t step = 0; step < STEPS_PER_KERNEL; step++) {
        if ((step & 0x3FF) == 0 && d_found) return;

        #pragma unroll
        for (int k = 0; k < KANGAROOS_PER_THREAD; k++) {
            uint32_t h = soa.is_tame[k] ? hash_point_tame_soa(&soa, k) 
                                          : hash_point_wild_soa(&soa, k);

            int j = select_jump(h);

            // Prefetch
            int prefetch_j = select_jump(h + 0x9E3779B9u);
            #if __CUDA_ARCH__ >= 700
                __builtin_prefetch(&s_jump_points[prefetch_j], 0, 3);
            #endif

            // Z=1 fast path for tame
            if (soa.is_tame[k] && soa_is_z1(&soa, k)) {
                point_add_mixed_Z1_mont_soa(&soa, k, &s_jump_points[j], &soa, k);
            } else {
                point_add_mixed_mont_soa(&soa, k, &s_jump_points[j], &soa, k);
            }

            // Negation map ONLY for tame
            if (soa.is_tame[k]) {
                soa_canonicalize_y_mont(&soa, k);
            }

            uint256_t jump_dist = {{0}};
            jump_dist.w[0] = s_jump_dist[j];
            uint256_t dist;
            soa_load_dist(&soa, k, &dist);
            add256_mod_n(&dist, &jump_dist, &dist);
            soa_store_dist(&soa, k, &dist);

            // Branchless loop detection
            uint32_t hash_match = (h == soa.last_hash[k]) ? 1 : 0;
            soa.loop_counter[k] = hash_match ? (soa.loop_counter[k] + 1) : 0;

            uint32_t need_restart = (soa.loop_counter[k] > LOOP_THRESHOLD) ? 1 : 0;
            if (__builtin_expect(need_restart, 0)) {
                if (!soa.is_tame[k]) {
                    rand135_range(&rng, &dist);
                    if (use_mod3) adjust_mod3(&dist, 2);
                    soa_store_dist(&soa, k, &dist);
                    uint64_t* xp = soa_x_ptr(&soa, k);
                    uint64_t* yp = soa_y_ptr(&soa, k);
                    uint64_t* zp = soa_z_ptr(&soa, k);
                    xp[0] = target.x.w[0]; xp[1] = target.x.w[1]; xp[2] = target.x.w[2]; xp[3] = target.x.w[3];
                    yp[0] = target.y.w[0]; yp[1] = target.y.w[1]; yp[2] = target.y.w[2]; yp[3] = target.y.w[3];
                    zp[0] = 1; zp[1] = 0; zp[2] = 0; zp[3] = 0;
                } else {
                    soa_to_jacobian(&soa, k, &T0x, &T0y);
                    uint64_t* dp = soa_dist_ptr(&soa, k);
                    dp[0] = thread_offset.w[0]; dp[1] = thread_offset.w[1];
                    dp[2] = thread_offset.w[2]; dp[3] = thread_offset.w[3];
                    if (k == 1) {
                        uint256_t d;
                        soa_load_dist(&soa, k, &d);
                        add256_mod_n(&d, &herd_spacing, &d);
                        soa_store_dist(&soa, k, &d);
                    }
                    if (use_mod3) {
                        uint256_t d;
                        soa_load_dist(&soa, k, &d);
                        adjust_mod3(&d, 0);
                        soa_store_dist(&soa, k, &d);
                    }
                }
                soa.loop_counter[k] = 0;
                atomicAdd((unsigned long long*)&d_loop_restarts, 1);
            }
            soa.last_hash[k] = h;

            // DP detection
            int is_dp = is_distinguished(h, dp_mask);
            unsigned dp_mask_warp = warp.ballot(is_dp);

            if (dp_mask_warp != 0) {
                uint32_t x_sig = x_signature_soa(&soa, k);
                uint32_t slot = compute_slot(h, dp_bits, dp_table_size);
                uint256_t dist;
                soa_load_dist(&soa, k, &dist);
                CompactDPEntry entry;
                int herd_id = soa.is_tame[k] ? 0 : (k - NUM_TAME);
                pack_dp(x_sig, &dist, soa.is_tame[k], herd_id, &entry);
                uint256_t result_k = {{0}};
                int collision_result = warp_collision_check_cg(warp, slot, &entry, &result_k);
                if (collision_result == 1) {
                    d_result_k = result_k;
                    atomicExch(&d_found, 1);
                    atomicAdd((unsigned long long*)&d_collision_count, 1);
                    return;
                }
            }
            warp_batch_steps++;
        }

        // Step counting
        if ((step & 0xFFFF) == 0) {
            uint64_t warp_sum = cg::reduce(warp, warp_batch_steps, cg::plus<uint64_t>());
            if (lane == 0) {
                atomicAdd((unsigned long long*)&d_total_steps, warp_sum);
            }
            warp_batch_steps = 0;
        }
    }

    if (warp_batch_steps > 0) {
        uint64_t warp_sum = cg::reduce(warp, warp_batch_steps, cg::plus<uint64_t>());
        if (lane == 0) {
            atomicAdd((unsigned long long*)&d_total_steps, warp_sum);
        }
    }
}

// ============================================================================
// DYNAMIC PARALLELISM — Adaptive kernel launch
// Launches child kernels for checkpoint save or DP table resize
// ============================================================================

#ifdef USE_DYNAMIC_PARALLELISM
__global__ void dp_resize_kernel(uint32_t new_dp_bits, uint32_t new_size) {
    // Zero out new DP table
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < new_size) {
        d_dp_table[idx].d0 = 0;
        d_dp_table[idx].metadata = 0;
    }
}

__global__ void checkpoint_save_kernel() {
    // Mark for host to save checkpoint
    // Host polls this and performs actual file I/O
    atomicExch(&d_found, 2); // 2 = checkpoint requested
}
#endif
