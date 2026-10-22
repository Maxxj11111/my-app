/**
 * PSCKangaroo v57.0.0 — Common definitions
 * Modular architecture: common.h + *.cuh + *.cpp
 */

#ifndef PSC_COMMON_H
#define PSC_COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <inttypes.h>

namespace cg = cooperative_groups;

// ============================================================================
// CONFIGURATION
// ============================================================================

#define MAX_GPUS 8
#define NUM_STREAMS_PER_GPU 4
#define NUM_WORDS_64 4
#define NUM_BITS 256

#define PUZZLE_NUMBER 135
#define RANGE_START_BITS (PUZZLE_NUMBER - 1)
#define RANGE_END_BITS PUZZLE_NUMBER

#define NUM_TAME 2
#define NUM_WILD 4
#define KANGAROOS_PER_THREAD (NUM_TAME + NUM_WILD)

#define NUM_JUMPS 256
#define JUMP_SELECT_BITS 8

#define DP_BITS_DEFAULT 23
#define DP_BITS_MIN 20
#define DP_BITS_MAX 26

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_SM 2
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)

#define STEPS_PER_KERNEL (1ULL << 20)
#define CHECKPOINT_INTERVAL_SEC 300
#define CHECKPOINT_FILE "psc_v57_checkpoint.dat"
#define CHECKPOINT_VERSION 0x00010057

#define MAX_TOTAL_THREADS (1U << 21)
#define LOOP_THRESHOLD 16384
#define IPC_SYNC_INTERVAL_SEC 60

// Pippenger window size
#define PIPPENGER_W 4
#define PIPPENGER_BUCKETS (1 << PIPPENGER_W)

// GLV window size for joint mult
#define GLV_WINDOW_SIZE 4
#define GLV_TABLE_SIZE (1 << GLV_WINDOW_SIZE)

// ============================================================================
// MONTGOMERY CONSTANTS
// ============================================================================

// p' = -p^(-1) mod 2^64 for secp256k1
#define MONT_P_PRIME 0xD838091DD2253531ULL

// ============================================================================
// SECP256K1 CONSTANTS (little-endian, 64-bit words)
// ============================================================================

static const uint64_t SECP_P_64[NUM_WORDS_64] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

static const uint64_t SECP_N_64[NUM_WORDS_64] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

static const uint64_t SECP_GX_64[NUM_WORDS_64] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};

static const uint64_t SECP_GY_64[NUM_WORDS_64] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
};

static const uint64_t POW2_134_64[NUM_WORDS_64] = {
    0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000040ULL, 0x0000000000000000ULL
};

static const uint64_t SECP_P_HALF_64[NUM_WORDS_64] = {
    0xFFFFFFFF7FFFFE17ULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
};

// GLV beta: endomorphism eigenvalue on curve
static const uint64_t GLV_BETA_64[NUM_WORDS_64] = {
    0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL
};

// GLV lambda: scalar such that lambda*G = beta*G
static const uint64_t GLV_LAMBDA_64[NUM_WORDS_64] = {
    0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
    0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL
};

static const uint64_t TWO_64[NUM_WORDS_64] = {
    0x2ULL, 0x0ULL, 0x0ULL, 0x0ULL
};

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

typedef struct { uint64_t w[NUM_WORDS_64]; } uint256_t;
typedef struct { uint64_t w[NUM_WORDS_64]; int neg; } int256_t;
typedef struct { uint256_t x; uint256_t y; } AffinePoint;
typedef struct { uint256_t x; uint256_t y; uint256_t z; } Point;

// SoA for register efficiency
typedef struct {
    uint64_t x[NUM_WORDS_64 * KANGAROOS_PER_THREAD];
    uint64_t y[NUM_WORDS_64 * KANGAROOS_PER_THREAD];
    uint64_t z[NUM_WORDS_64 * KANGAROOS_PER_THREAD];
    uint64_t dist[NUM_WORDS_64 * KANGAROOS_PER_THREAD];
    uint32_t last_hash[KANGAROOS_PER_THREAD];
    uint32_t loop_counter[KANGAROOS_PER_THREAD];
    uint8_t  is_tame[KANGAROOS_PER_THREAD];
} KangarooSoA;

typedef struct { 
    uint64_t d0;
    uint64_t d1;
    uint64_t d2;
    uint32_t metadata;
    uint32_t padding;
} CompactDPEntry;

// Host types
typedef struct { uint64_t w[NUM_WORDS_64]; } h_uint256_t;
typedef struct { h_uint256_t x; h_uint256_t y; } h_AffinePoint;
typedef struct { h_uint256_t x; h_uint256_t y; h_uint256_t z; } h_Point;

// ============================================================================
// DEVICE CONSTANTS (extern, defined in kernel.cu)
// ============================================================================

extern __constant__ uint64_t d_p[NUM_WORDS_64];
extern __constant__ uint64_t d_n[NUM_WORDS_64];
extern __constant__ uint64_t d_Gx[NUM_WORDS_64];
extern __constant__ uint64_t d_Gy[NUM_WORDS_64];
extern __constant__ uint64_t d_T0x[NUM_WORDS_64];
extern __constant__ uint64_t d_T0y[NUM_WORDS_64];
extern __constant__ uint64_t d_p_half[NUM_WORDS_64];
extern __constant__ uint64_t d_beta[NUM_WORDS_64];
extern __constant__ uint64_t d_lambda[NUM_WORDS_64];
extern __constant__ uint64_t d_two[NUM_WORDS_64];

extern __constant__ AffinePoint c_jump_points[NUM_JUMPS];
extern __constant__ uint64_t c_jump_dist[NUM_JUMPS];

// GLV tables: G1=G, G2=lambda*G, both in Montgomery domain
extern __constant__ AffinePoint c_glv_G1;
extern __constant__ AffinePoint c_glv_G2;

extern __device__ CompactDPEntry* d_dp_table;
extern __device__ uint256_t d_result_k;
extern __device__ uint32_t d_found;
extern __device__ uint64_t d_total_steps;
extern __device__ uint64_t d_collision_count;
extern __device__ uint64_t d_dp_insertions;
extern __device__ uint64_t d_loop_restarts;
extern __device__ uint32_t d_current_dp_bits;
extern __device__ uint32_t d_dp_table_size;

// Global state
extern int g_use_mod3_filter;
extern int g_num_gpus;

// ============================================================================
// INLINE DEVICE MATH (used across multiple .cuh files)
// ============================================================================

__device__ __forceinline__ uint32_t add256_v2(const uint256_t* a, const uint256_t* b, uint256_t* r) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint64_t sum = a->w[i] + b->w[i] + carry;
        r->w[i] = sum;
        carry = (sum < a->w[i]) | ((sum == a->w[i]) & carry);
    }
    return (uint32_t)carry;
}

__device__ __forceinline__ uint32_t sub256(const uint256_t* a, const uint256_t* b, uint256_t* r) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint64_t diff = a->w[i] - b->w[i] - borrow;
        r->w[i] = diff;
        borrow = (a->w[i] < b->w[i]) | ((a->w[i] == b->w[i]) & borrow);
    }
    return (uint32_t)borrow;
}

__device__ __forceinline__ int cmp256(const uint256_t* a, const uint256_t* b) {
    #pragma unroll
    for (int i = NUM_WORDS_64 - 1; i >= 0; i--) {
        if (a->w[i] > b->w[i]) return 1;
        if (a->w[i] < b->w[i]) return -1;
    }
    return 0;
}

// ============================================================================
// HOST INLINE MATH
// ============================================================================

static inline uint32_t h_add256(const h_uint256_t* a, const h_uint256_t* b, h_uint256_t* r) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < NUM_WORDS_64; i++) {
        unsigned __int128 sum = (unsigned __int128)a->w[i] + b->w[i] + (uint64_t)carry;
        r->w[i] = (uint64_t)sum;
        carry = sum >> 64;
    }
    return (uint32_t)carry;
}

static inline uint32_t h_sub256(const h_uint256_t* a, const h_uint256_t* b, h_uint256_t* r) {
    __int128 borrow = 0;
    for (int i = 0; i < NUM_WORDS_64; i++) {
        __int128 diff = (__int128)a->w[i] - b->w[i] - (uint64_t)borrow;
        r->w[i] = (uint64_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    return (uint32_t)borrow;
}

static inline int h_cmp256(const h_uint256_t* a, const h_uint256_t* b) {
    for (int i = NUM_WORDS_64 - 1; i >= 0; i--) {
        if (a->w[i] > b->w[i]) return 1;
        if (a->w[i] < b->w[i]) return -1;
    }
    return 0;
}


// Forward declarations (defined in .cpp files)
void precompute_jumps();
void precompute_t0();
void precompute_glv_tables();
void init_dp_table_multi_gpu(int dp_bits, int num_gpus);
void sync_dp_table_between_gpus(int num_gpus);
void sync_dp_table_nvlink(int num_gpus);
int calculate_adaptive_dp_bits(uint64_t total_steps, uint64_t dp_insertions, int current_dp_bits);
void save_checkpoint(int dp_bits, int num_gpus);
int load_checkpoint(int* dp_bits, int num_gpus);
int verify_key(const h_uint256_t* diff, const h_AffinePoint* target);
void benchmark_v57(int num_gpus);
void hex_to_uint256(const char* hex, h_uint256_t* out);
void print_usage(const char* prog);

// Kernel forward declaration (defined in kernel.cu)
__global__ void kangaroo_kernel_v57(Point target, uint64_t seed, uint32_t dp_bits, uint32_t dp_table_size, int use_mod3);

#endif // PSC_COMMON_H
