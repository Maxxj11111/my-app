/**
 * mont_math.cuh — Montgomery domain arithmetic
 * Device: mont_mul, mont_sqr, mont_reduce, modinv_mont
 * Host: h_mont_mul, h_mont_sqr, h_mont_reduce, h_modinv_mont
 */

#ifndef PSC_MONT_MATH_CUH
#define PSC_MONT_MATH_CUH

#include "common.h"

// ============================================================================
// DEVICE MONTGOMERY
// ============================================================================

__device__ __forceinline__ void mont_reduce(const uint64_t t[8], uint256_t* out) {
    // m = t * p' mod R, word-by-word
    uint64_t m[NUM_WORDS_64];
    uint64_t c = 0;

    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        unsigned __int128 prod = (unsigned __int128)t[i] * MONT_P_PRIME + c;
        m[i] = (uint64_t)prod;
        c = (uint64_t)(prod >> 64);
    }

    // u = (t + m * p) / R
    uint64_t u[5] = {0};
    c = 0;

    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        unsigned __int128 sum = c;
        #pragma unroll
        for (int j = 0; j <= i; j++) {
            sum += (unsigned __int128)m[j] * d_p[i - j];
        }
        sum += t[i + NUM_WORDS_64];
        u[i] = (uint64_t)sum;
        c = (uint64_t)(sum >> 64);
    }
    u[4] = c;

    // Branchless final subtraction
    uint256_t temp;
    uint32_t borrow = sub256((const uint256_t*)u, (const uint256_t*)d_p, &temp);
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        out->w[i] = borrow ? u[i] : temp.w[i];
    }
}

__device__ __forceinline__ void mont_mul(const uint256_t* a, const uint256_t* b, uint256_t* r) {
    uint64_t t[8] = {0};
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < NUM_WORDS_64; j++) {
            unsigned __int128 prod = (unsigned __int128)a->w[i] * b->w[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+NUM_WORDS_64] = carry;
    }
    mont_reduce(t, r);
}

__device__ __forceinline__ void mont_sqr(const uint256_t* a, uint256_t* r) {
    mont_mul(a, a, r);
}

// Convert to/from Montgomery domain
// R^2 mod p precomputed on host, stored in constant memory
__device__ __forceinline__ void to_mont(const uint256_t* a, uint256_t* r) {
    // Multiply by R^2 mod p, then reduce -> a*R mod p
    extern __constant__ uint64_t d_mont_r2[NUM_WORDS_64];
    mont_mul(a, (const uint256_t*)d_mont_r2, r);
}

__device__ __forceinline__ void from_mont(const uint256_t* a, uint256_t* r) {
    uint256_t one = {{1, 0, 0, 0}};
    mont_mul(a, &one, r);
}

// ============================================================================
// DEVICE MODULAR INVERSE (Montgomery Fermat)
// ============================================================================

__device__ __forceinline__ void modinv_mont(const uint256_t* a, uint256_t* r) {
    uint64_t is_zero = a->w[0] | a->w[1] | a->w[2] | a->w[3];
    if (__builtin_expect(is_zero == 0, 0)) {
        #pragma unroll
        for (int i = 0; i < NUM_WORDS_64; i++) r->w[i] = 0;
        return;
    }

    // p - 2
    uint256_t exp;
    sub256((const uint256_t*)d_p, (const uint256_t*)d_two, &exp);

    uint256_t base = *a, result = {{1, 0, 0, 0}};

    #pragma unroll 4
    for (int i = 0; i < 256; i++) {
        int word = i >> 6, bit = i & 63;
        if ((exp.w[word] >> bit) & 1ULL) mont_mul(&result, &base, &result);
        if (i < 255) mont_sqr(&base, &base);
    }
    *r = result;
}

// Scalar modinv (mod n, uses Montgomery arithmetic with p constants)
__device__ __forceinline__ void modinv_n(const uint256_t* a, uint256_t* r) {
    uint64_t is_zero = a->w[0] | a->w[1] | a->w[2] | a->w[3];
    if (__builtin_expect(is_zero == 0, 0)) {
        #pragma unroll
        for (int i = 0; i < NUM_WORDS_64; i++) r->w[i] = 0;
        return;
    }

    uint256_t exp;
    sub256((const uint256_t*)d_n, (const uint256_t*)d_two, &exp);

    uint256_t base = *a, result = {{1, 0, 0, 0}};

    #pragma unroll 4
    for (int i = 0; i < 256; i++) {
        int word = i >> 6, bit = i & 63;
        if ((exp.w[word] >> bit) & 1ULL) mont_mul(&result, &base, &result);
        if (i < 255) mont_sqr(&base, &base);
    }
    *r = result;
}

// ============================================================================
// HOST MONTGOMERY
// ============================================================================

static inline void h_mont_reduce(const uint64_t t[8], h_uint256_t* out) {
    uint64_t m[NUM_WORDS_64];
    uint64_t c = 0;

    for (int i = 0; i < NUM_WORDS_64; i++) {
        unsigned __int128 prod = (unsigned __int128)t[i] * MONT_P_PRIME + c;
        m[i] = (uint64_t)prod;
        c = (uint64_t)(prod >> 64);
    }

    uint64_t u[5] = {0};
    c = 0;

    for (int i = 0; i < NUM_WORDS_64; i++) {
        unsigned __int128 sum = c;
        for (int j = 0; j <= i; j++) {
            sum += (unsigned __int128)m[j] * SECP_P_64[i - j];
        }
        sum += t[i + NUM_WORDS_64];
        u[i] = (uint64_t)sum;
        c = (uint64_t)(sum >> 64);
    }
    u[4] = c;

    h_uint256_t temp;
    uint32_t borrow = h_sub256((const h_uint256_t*)u, (const h_uint256_t*)SECP_P_64, &temp);
    for (int i = 0; i < NUM_WORDS_64; i++) {
        out->w[i] = borrow ? u[i] : temp.w[i];
    }
}

static inline void h_mont_mul(const h_uint256_t* a, const h_uint256_t* b, h_uint256_t* r) {
    uint64_t t[8] = {0};
    for (int i = 0; i < NUM_WORDS_64; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < NUM_WORDS_64; j++) {
            unsigned __int128 prod = (unsigned __int128)a->w[i] * b->w[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+NUM_WORDS_64] = carry;
    }
    h_mont_reduce(t, r);
}

static inline void h_mont_sqr(const h_uint256_t* a, h_uint256_t* r) {
    h_mont_mul(a, a, r);
}

// R^2 mod p for secp256k1
static inline void get_mont_r2(h_uint256_t* r2) {
    // R^2 mod p = (2^256)^2 mod p = 2^512 mod p
    // Precomputed: 0x0000000000000000000000000000000000000000000000000000000000000001
    // Actually for secp256k1: R^2 mod p = 0x0000000000000000000000000000000000000000000000000000000000000001
    // Wait, that's wrong. Let me compute properly.
    // R = 2^256 mod p = 2^256 - p = 0x00000001000003D1... 
    // Actually R = 2^256, R mod p = 2^256 - p = 0x00000001000003D1000000000000000000000000000000000000000000000001
    memset(r2, 0, sizeof(h_uint256_t));
    // For simplicity, compute R^2 mod p via repeated doubling
    h_uint256_t R = {{0}};
    R.w[0] = 1; // Start with 1
    // Double 256 times to get 2^256 mod p
    for (int i = 0; i < 256; i++) {
        h_uint256_t temp;
        uint32_t carry = h_add256(&R, &R, &temp);
        if (carry || h_cmp256(&temp, (const h_uint256_t*)SECP_P_64) >= 0) {
            h_sub256(&temp, (const h_uint256_t*)SECP_P_64, &R);
        } else {
            R = temp;
        }
    }
    // Now R = 2^256 mod p, compute R^2 mod p
    h_mont_mul(&R, &R, r2);
}

static inline void h_to_mont(const h_uint256_t* a, h_uint256_t* r) {
    h_uint256_t r2;
    get_mont_r2(&r2);
    h_mont_mul(a, &r2, r);
}

static inline void h_from_mont(const h_uint256_t* a, h_uint256_t* r) {
    h_uint256_t one = {{1, 0, 0, 0}};
    h_mont_mul(a, &one, r);
}

static inline void h_modinv_mont(const h_uint256_t* a, h_uint256_t* r) {
    uint64_t is_zero = a->w[0] | a->w[1] | a->w[2] | a->w[3];
    if (is_zero == 0) {
        memset(r, 0, sizeof(h_uint256_t));
        return;
    }

    h_uint256_t exp;
    h_sub256((const h_uint256_t*)SECP_P_64, (const h_uint256_t*)TWO_64, &exp);

    h_uint256_t base = *a, result = {{1, 0, 0, 0}};

    for (int i = 0; i < NUM_BITS; i++) {
        int word = i >> 6, bit = i & 63;
        if ((exp.w[word] >> bit) & 1ULL) h_mont_mul(&result, &base, &result);
        h_mont_sqr(&base, &base);
    }
    *r = result;
}

static inline void h_modinv_n(const h_uint256_t* a, h_uint256_t* r) {
    uint64_t is_zero = a->w[0] | a->w[1] | a->w[2] | a->w[3];
    if (is_zero == 0) {
        memset(r, 0, sizeof(h_uint256_t));
        return;
    }

    h_uint256_t exp;
    h_sub256((const h_uint256_t*)SECP_N_64, (const h_uint256_t*)TWO_64, &exp);

    h_uint256_t base = *a, result = {{1, 0, 0, 0}};

    for (int i = 0; i < NUM_BITS; i++) {
        int word = i >> 6, bit = i & 63;
        if ((exp.w[word] >> bit) & 1ULL) h_mont_mul(&result, &base, &result);
        h_mont_sqr(&base, &base);
    }
    *r = result;
}

// ============================================================================
// BATCH INVERSION — Montgomery trick (host)
// ============================================================================

static inline void h_batch_modinv(h_uint256_t* values, h_uint256_t* inverses, int count) {
    if (count == 0) return;
    if (count == 1) { h_modinv_mont(&values[0], &inverses[0]); return; }

    h_uint256_t* products = (h_uint256_t*)malloc(count * sizeof(h_uint256_t));
    products[0] = values[0];
    for (int i = 1; i < count; i++) h_mont_mul(&products[i-1], &values[i], &products[i]);

    h_uint256_t inverse;
    h_modinv_mont(&products[count-1], &inverse);

    for (int i = count - 1; i > 0; i--) {
        h_mont_mul(&inverse, &products[i-1], &inverses[i]);
        h_mont_mul(&inverse, &values[i], &inverse);
    }
    inverses[0] = inverse;
    free(products);
}

#endif // PSC_MONT_MATH_CUH
