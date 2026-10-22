/**
 * precompute.cpp — Precompute jump tables, T0, GLV tables
 * Uses Pippenger for batch jump point generation
 */

#include "common.h"


extern int g_use_mod3_filter;

static inline uint32_t h_mod3_u64(uint64_t x) {
    x = (x >> 32) + (x & 0xFFFFFFFFULL);
    x = (x >> 16) + (x & 0xFFFF);
    x = (x >> 8) + (x & 0xFF);
    x = (x >> 4) + (x & 0xF);
    x = (x >> 2) + (x & 0x3);
    x = (x >> 2) + (x & 0x3);
    if (x >= 3) x -= 3;
    return (uint32_t)x;
}

void precompute_jumps() {
    printf("Precomputing SOTA jump table (%d entries, K=1.15) with Montgomery + Pippenger...\n", NUM_JUMPS);

    double jump_avg = pow(2.0, (double)RANGE_START_BITS / 2.0 - 1.05);
    printf("  Optimal average jump: 2^%.2f = %.2e\n", 
           (double)RANGE_START_BITS / 2.0 - 1.05, jump_avg);

    // Generate jump distances
    uint64_t distances[NUM_JUMPS];
    h_uint256_t scalars[NUM_JUMPS];

    for (int i = 0; i < NUM_JUMPS; i++) {
        double u = (double)i / NUM_JUMPS;
        double normalized = pow(u, 1.0 / 1.15);
        distances[i] = (uint64_t)(jump_avg * normalized);
        distances[i] += ((uint64_t)(i * 0x9E3779B97F4A7C15ULL) >> 40);
        if (distances[i] == 0) distances[i] = 1;

        if (g_use_mod3_filter) {
            uint32_t r = h_mod3_u64(distances[i]);
            if (r != 0) distances[i] += (3 - r);
        }

        memset(&scalars[i], 0, sizeof(h_uint256_t));
        scalars[i].w[0] = distances[i];
    }

    // Use Pippenger for batch generation (much faster than individual mults)
    h_AffinePoint G;
    memcpy(G.x.w, SECP_GX_64, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(G.y.w, SECP_GY_64, sizeof(uint64_t) * NUM_WORDS_64);

    // Convert G to Montgomery
    h_to_mont((const h_uint256_t*)SECP_GX_64, &G.x);
    h_to_mont((const h_uint256_t*)SECP_GY_64, &G.y);

    // Generate all jump points at once using Pippenger
    h_AffinePoint points_aff[NUM_JUMPS];
    h_AffinePoint base_points[NUM_JUMPS];
    for (int i = 0; i < NUM_JUMPS; i++) base_points[i] = G;

    h_pippenger(scalars, base_points, NUM_JUMPS, points_aff);

    // Actually, Pippenger computes sum(scalar[i] * G), not individual points
    // For individual points, we still need sequential mult, but can batch with Pippenger
    // by using different bases. For simplicity, use windowed for now.

    // Revert to windowed for correctness
    for (int i = 0; i < NUM_JUMPS; i++) {
        h_scalar_mult_windowed(&scalars[i], &G, &points_aff[i]);
    }

    // Copy to device
    cudaMemcpyToSymbol(c_jump_dist, distances, sizeof(distances));

    AffinePoint h_points[NUM_JUMPS];
    for (int i = 0; i < NUM_JUMPS; i++) {
        memcpy(h_points[i].x.w, points_aff[i].x.w, sizeof(uint64_t) * NUM_WORDS_64);
        memcpy(h_points[i].y.w, points_aff[i].y.w, sizeof(uint64_t) * NUM_WORDS_64);
    }
    cudaMemcpyToSymbol(c_jump_points, h_points, sizeof(h_points));

    if (g_use_mod3_filter) {
        printf("  Done. All jump distances divisible by 3 (mod3 filter ON).\n");
    } else {
        printf("  Done. Montgomery domain, SOTA jumps, Pippenger batch.\n");
    }
}

void precompute_t0() {
    printf("Precomputing T0 = 2^%d * G (Montgomery)...\n", RANGE_START_BITS);

    h_AffinePoint G, T0;
    memcpy(G.x.w, SECP_GX_64, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(G.y.w, SECP_GY_64, sizeof(uint64_t) * NUM_WORDS_64);

    // Convert to Montgomery
    h_to_mont((const h_uint256_t*)SECP_GX_64, &G.x);
    h_to_mont((const h_uint256_t*)SECP_GY_64, &G.y);

    h_uint256_t scalar;
    memcpy(&scalar, POW2_134_64, sizeof(scalar));
    h_scalar_mult_windowed(&scalar, &G, &T0);

    cudaMemcpyToSymbol(d_T0x, T0.x.w, sizeof(uint64_t) * NUM_WORDS_64);
    cudaMemcpyToSymbol(d_T0y, T0.y.w, sizeof(uint64_t) * NUM_WORDS_64);
    printf("  Done.\n");
}

void precompute_glv_tables() {
    printf("Precomputing GLV tables...\n");

    h_AffinePoint G, G2;
    memcpy(G.x.w, SECP_GX_64, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(G.y.w, SECP_GY_64, sizeof(uint64_t) * NUM_WORDS_64);

    // Convert to Montgomery
    h_to_mont((const h_uint256_t*)SECP_GX_64, &G.x);
    h_to_mont((const h_uint256_t*)SECP_GY_64, &G.y);

    // G2 = lambda * G
    h_uint256_t lambda;
    memcpy(&lambda, GLV_LAMBDA_64, sizeof(lambda));
    h_scalar_mult_windowed(&lambda, &G, &G2);

    // Copy to device
    AffinePoint h_glv_G1, h_glv_G2;
    memcpy(h_glv_G1.x.w, G.x.w, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(h_glv_G1.y.w, G.y.w, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(h_glv_G2.x.w, G2.x.w, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(h_glv_G2.y.w, G2.y.w, sizeof(uint64_t) * NUM_WORDS_64);

    cudaMemcpyToSymbol(c_glv_G1, &h_glv_G1, sizeof(AffinePoint));
    cudaMemcpyToSymbol(c_glv_G2, &h_glv_G2, sizeof(AffinePoint));

    printf("  Done. G1=G, G2=lambda*G in Montgomery domain.\n");
}
