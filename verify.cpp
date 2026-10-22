/**
 * verify.cpp — Key verification with multiple possibilities
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

static inline int h_mod3_u256(const h_uint256_t* a) {
    uint32_t sum = 0;
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint32_t w = h_mod3_u64(a->w[i]);
        sum += w;
        if (sum >= 3) sum -= 3;
    }
    return (int)sum;
}

static inline void h_mod_n(const h_uint256_t* a, h_uint256_t* r) {
    *r = *a;
    h_uint256_t temp;
    while (h_cmp256(r, (const h_uint256_t*)SECP_N_64) >= 0) {
        h_sub256(r, (const h_uint256_t*)SECP_N_64, &temp);
        *r = temp;
    }
}

int verify_key(const h_uint256_t* diff, const h_AffinePoint* target) {
    printf("Verifying key...\n");

    h_AffinePoint G, result;
    memcpy(G.x.w, SECP_GX_64, sizeof(uint64_t) * NUM_WORDS_64);
    memcpy(G.y.w, SECP_GY_64, sizeof(uint64_t) * NUM_WORDS_64);

    // Convert to Montgomery
    h_to_mont((const h_uint256_t*)SECP_GX_64, &G.x);
    h_to_mont((const h_uint256_t*)SECP_GY_64, &G.y);

    h_uint256_t pow2_134;
    memcpy(&pow2_134, POW2_134_64, sizeof(pow2_134));

    // Possibility 1: k = 2^134 + diff (mod n)
    h_uint256_t k_candidate;
    uint32_t carry = h_add256(&pow2_134, diff, &k_candidate);
    h_uint256_t temp;
    while (carry || h_cmp256(&k_candidate, (const h_uint256_t*)SECP_N_64) >= 0) {
        h_sub256(&k_candidate, (const h_uint256_t*)SECP_N_64, &temp);
        k_candidate = temp;
        carry = 0;
    }

    int mod3 = h_mod3_u256(&k_candidate);
    if (!g_use_mod3_filter || mod3 == 2) {
        h_scalar_mult_windowed(&k_candidate, &G, &result);
        h_from_mont(&result.x, &result.x);
        h_from_mont(&result.y, &result.y);
        if (memcmp(result.x.w, target->x.w, sizeof(uint64_t) * NUM_WORDS_64) == 0 &&
            memcmp(result.y.w, target->y.w, sizeof(uint64_t) * NUM_WORDS_64) == 0) {
            printf("  VERIFICATION PASSED! (k = 2^134 + diff");
            if (g_use_mod3_filter) printf(", k ≡ 2 mod 3");
            printf(")\n  k = ");
            for (int i = NUM_WORDS_64 - 1; i >= 0; i--) printf("%016lx", k_candidate.w[i]);
            printf("\n");
            return 1;
        }
    } else if (g_use_mod3_filter) {
        printf("  Skipping k = 2^134 + diff (k mod 3 = %d, expected 2)\n", mod3);
    }

    // Possibility 2: k = 2^134 - diff (mod n)
    if (h_cmp256(&pow2_134, diff) >= 0) {
        h_sub256(&pow2_134, diff, &k_candidate);
    } else {
        h_uint256_t temp;
        h_sub256(diff, &pow2_134, &temp);
        h_sub256((const h_uint256_t*)SECP_N_64, &temp, &k_candidate);
    }
    h_mod_n(&k_candidate, &k_candidate);

    mod3 = h_mod3_u256(&k_candidate);
    if (!g_use_mod3_filter || mod3 == 2) {
        h_scalar_mult_windowed(&k_candidate, &G, &result);
        h_from_mont(&result.x, &result.x);
        h_from_mont(&result.y, &result.y);
        if (memcmp(result.x.w, target->x.w, sizeof(uint64_t) * NUM_WORDS_64) == 0 &&
            memcmp(result.y.w, target->y.w, sizeof(uint64_t) * NUM_WORDS_64) == 0) {
            printf("  VERIFICATION PASSED! (k = 2^134 - diff");
            if (g_use_mod3_filter) printf(", k ≡ 2 mod 3");
            printf(")\n  k = ");
            for (int i = NUM_WORDS_64 - 1; i >= 0; i--) printf("%016lx", k_candidate.w[i]);
            printf("\n");
            return 1;
        }
    } else if (g_use_mod3_filter) {
        printf("  Skipping k = 2^134 - diff (k mod 3 = %d, expected 2)\n", mod3);
    }

    // Possibility 3: k = diff
    mod3 = h_mod3_u256(diff);
    if (!g_use_mod3_filter || mod3 == 2) {
        h_scalar_mult_windowed(diff, &G, &result);
        h_from_mont(&result.x, &result.x);
        h_from_mont(&result.y, &result.y);
        if (memcmp(result.x.w, target->x.w, sizeof(uint64_t) * NUM_WORDS_64) == 0 &&
            memcmp(result.y.w, target->y.w, sizeof(uint64_t) * NUM_WORDS_64) == 0) {
            printf("  VERIFICATION PASSED! (k = diff");
            if (g_use_mod3_filter) printf(", k ≡ 2 mod 3");
            printf(")\n");
            return 1;
        }
    } else if (g_use_mod3_filter) {
        printf("  Skipping k = diff (k mod 3 = %d, expected 2)\n", mod3);
    }

    // Possibility 4: k = n - diff
    h_uint256_t n_minus_diff;
    h_sub256((const h_uint256_t*)SECP_N_64, diff, &n_minus_diff);
    h_mod_n(&n_minus_diff, &k_candidate);
    mod3 = h_mod3_u256(&k_candidate);
    if (!g_use_mod3_filter || mod3 == 2) {
        h_scalar_mult_windowed(&n_minus_diff, &G, &result);
        h_from_mont(&result.x, &result.x);
        h_from_mont(&result.y, &result.y);
        if (memcmp(result.x.w, target->x.w, sizeof(uint64_t) * NUM_WORDS_64) == 0 &&
            memcmp(result.y.w, target->y.w, sizeof(uint64_t) * NUM_WORDS_64) == 0) {
            printf("  VERIFICATION PASSED! (k = n - diff");
            if (g_use_mod3_filter) printf(", k ≡ 2 mod 3");
            printf(")\n");
            return 1;
        }
    } else if (g_use_mod3_filter) {
        printf("  Skipping k = n - diff (k mod 3 = %d, expected 2)\n", mod3);
    }

    printf("  VERIFICATION FAILED!\n");
    return 0;
}
