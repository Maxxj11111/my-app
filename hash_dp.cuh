/**
 * hash_dp.cuh — Hash functions, DP pack/unpack, collision detection
 * - xxHash32 for point hashing
 * - Compact 24-byte DP entries
 * - Cooperative Groups collision check with atomicCAS
 */

#ifndef PSC_HASH_DP_CUH
#define PSC_HASH_DP_CUH

#include "common.h"

#define XXH_PRIME32_1 0x9E3779B1U
#define XXH_PRIME32_2 0x85EBCA77U
#define XXH_PRIME32_3 0xC2B2AE3DU
#define XXH_PRIME32_4 0x27D4EB2FU
#define XXH_PRIME32_5 0x165667B1U

__device__ __forceinline__ uint32_t xxh32_rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// Tame hash: only X coordinate (with negation map, Y is canonicalized)
__device__ __forceinline__ uint32_t hash_point_tame_soa(const KangarooSoA* soa, int k) {
    const uint64_t* x_ptr = &soa->x[k * NUM_WORDS_64];
    uint32_t h = XXH_PRIME32_5 + 32;
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint64_t w = x_ptr[i];
        uint32_t lo = (uint32_t)w;
        uint32_t hi = (uint32_t)(w >> 32);
        h += lo * XXH_PRIME32_3;
        h = xxh32_rotl(h, 17) * XXH_PRIME32_4;
        h += hi * XXH_PRIME32_3;
        h = xxh32_rotl(h, 17) * XXH_PRIME32_4;
    }
    h ^= h >> 15; h *= XXH_PRIME32_2; h ^= h >> 13; h *= XXH_PRIME32_3; h ^= h >> 16;
    return h;
}

// Wild hash: X + Y (no negation map, need full point signature)
__device__ __forceinline__ uint32_t hash_point_wild_soa(const KangarooSoA* soa, int k) {
    const uint64_t* x_ptr = &soa->x[k * NUM_WORDS_64];
    const uint64_t* y_ptr = &soa->y[k * NUM_WORDS_64];
    uint32_t h = XXH_PRIME32_5 + 64;
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint64_t xw = x_ptr[i];
        uint64_t yw = y_ptr[i];
        uint32_t xlo = (uint32_t)xw, xhi = (uint32_t)(xw >> 32);
        uint32_t ylo = (uint32_t)yw, yhi = (uint32_t)(yw >> 32);
        h += xlo * XXH_PRIME32_3; h = xxh32_rotl(h, 17) * XXH_PRIME32_4;
        h += xhi * XXH_PRIME32_3; h = xxh32_rotl(h, 17) * XXH_PRIME32_4;
        h += ylo * XXH_PRIME32_3; h = xxh32_rotl(h, 17) * XXH_PRIME32_4;
        h += yhi * XXH_PRIME32_3; h = xxh32_rotl(h, 17) * XXH_PRIME32_4;
    }
    h ^= h >> 15; h *= XXH_PRIME32_2; h ^= h >> 13; h *= XXH_PRIME32_3; h ^= h >> 16;
    return h;
}

__device__ __forceinline__ int select_jump(uint32_t h) {
    return (h >> (32 - JUMP_SELECT_BITS)) & (NUM_JUMPS - 1);
}

__device__ __forceinline__ int is_distinguished(uint32_t h, uint32_t dp_mask) {
    return (h & dp_mask) == 0;
}

__device__ __forceinline__ uint32_t x_signature_soa(const KangarooSoA* soa, int k) {
    const uint64_t* x_ptr = &soa->x[k * NUM_WORDS_64];
    uint32_t sig = (uint32_t)(x_ptr[3] >> 32) ^ (uint32_t)(x_ptr[3]);
    sig ^= (uint32_t)(x_ptr[2] >> 32) ^ (uint32_t)(x_ptr[2]);
    sig ^= (uint32_t)(x_ptr[1] >> 32) ^ (uint32_t)(x_ptr[1]);
    sig ^= (uint32_t)(x_ptr[0] >> 32) ^ (uint32_t)(x_ptr[0]);
    return sig;
}

__device__ __forceinline__ uint32_t compute_slot(uint32_t h, uint32_t dp_bits, uint32_t dp_table_size) {
    uint32_t slot = h >> dp_bits;
    slot ^= h >> (dp_bits + 8 < 32 ? dp_bits + 8 : 24);
    slot ^= h * 0x9E3779B9u;
    return slot & (dp_table_size - 1);
}

// ============================================================================
// DP PACK/UNPACK — 24 bytes per entry
// ============================================================================

__device__ __forceinline__ void pack_dp(uint32_t x_sig, const uint256_t* dist,
                                        int is_tame, int herd,
                                        CompactDPEntry* entry) {
    entry->d0 = dist->w[0];
    entry->d1 = dist->w[1];
    uint32_t meta = (x_sig & 0xFFFFF)
                  | ((is_tame & 1) << 20)
                  | ((herd & 3) << 21)
                  | (1u << 31);
    entry->d2 = ((uint64_t)meta << 32) | (dist->w[2] & 0xFFFFFFFFULL);
    entry->metadata = meta;
    entry->padding = 0;
}

__device__ __forceinline__ void unpack_dp(const CompactDPEntry* entry,
                                            uint32_t* x_sig, uint256_t* dist,
                                            int* is_tame, int* herd, int* valid) {
    dist->w[0] = entry->d0;
    dist->w[1] = entry->d1;
    dist->w[2] = entry->d2;
    dist->w[3] = 0;
    *x_sig = entry->metadata & 0xFFFFF;
    *is_tame = (entry->metadata >> 20) & 1;
    *herd = (entry->metadata >> 21) & 3;
    *valid = (entry->metadata >> 31) & 1;
}

// ============================================================================
// COLLISION CHECK — Cooperative Groups, warp-level
// ============================================================================

__device__ __forceinline__ int warp_collision_check_cg(cg::thread_block_tile<32> warp,
                                                        uint32_t slot,
                                                        CompactDPEntry* entry,
                                                        uint256_t* result_k_out) {
    int lane = warp.thread_rank();
    int collision = 0;

    // Atomic insert: CAS d0 from 0 to our value
    unsigned long long expected = 0ULL;
    unsigned long long new_d0 = entry->d0;
    unsigned long long old_d0 = atomicCAS(
        (unsigned long long*)&d_dp_table[slot].d0, expected, new_d0);

    if (old_d0 == 0) {
        // Empty slot: write rest of entry
        atomicExch((unsigned long long*)&d_dp_table[slot].d1, entry->d1);
        atomicExch((unsigned long long*)&d_dp_table[slot].d2, entry->d2);
        atomicExch((unsigned int*)&d_dp_table[slot].metadata, 
                   (unsigned int)(entry->d2 >> 32));
        __threadfence();
        atomicAdd((unsigned long long*)&d_dp_insertions, 1);
        return 0;
    }

    // Slot occupied: read full entry (spin until valid)
    CompactDPEntry old_entry;
    int spins = 0;
    do {
        old_entry.metadata = d_dp_table[slot].metadata;
        if ((old_entry.metadata >> 31) & 1) break;
        if (++spins > 1000) return 0;
    } while (1);

    __threadfence();
    old_entry.d0 = d_dp_table[slot].d0;
    old_entry.d1 = d_dp_table[slot].d1;
    old_entry.d2 = d_dp_table[slot].d2;

    uint32_t old_x_sig, new_x_sig;
    uint256_t old_dist, new_dist, diff;
    int old_is_tame, new_is_tame;
    int old_herd, new_herd;
    int old_valid, new_valid;

    unpack_dp(&old_entry, &old_x_sig, &old_dist, &old_is_tame, &old_herd, &old_valid);
    unpack_dp(entry, &new_x_sig, &new_dist, &new_is_tame, &new_herd, &new_valid);

    if (old_x_sig == new_x_sig && old_valid && new_valid && old_is_tame != new_is_tame) {
        if (cmp256(&new_dist, &old_dist) >= 0) {
            sub256_mod_n(&new_dist, &old_dist, &diff);
        } else {
            sub256_mod_n(&old_dist, &new_dist, &diff);
        }
        *result_k_out = diff;
        collision = 1;
    }

    // Broadcast collision result within warp
    unsigned collision_mask = warp.ballot(collision);
    if (collision_mask != 0) {
        int winner = __ffs(collision_mask) - 1;
        result_k_out->w[0] = warp.shfl(result_k_out->w[0], winner);
        result_k_out->w[1] = warp.shfl(result_k_out->w[1], winner);
        result_k_out->w[2] = warp.shfl(result_k_out->w[2], winner);
        result_k_out->w[3] = warp.shfl(result_k_out->w[3], winner);
        return (lane == winner) ? 1 : -1;
    }
    return 0;
}

#endif // PSC_HASH_DP_CUH
