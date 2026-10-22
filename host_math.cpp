/**
 * host_math.cpp — Host-side elliptic curve math
 * - Pippenger multi-scalar multiplication
 * - Windowed scalar multiplication
 * - Batch inversion (Montgomery trick)
 * - GLV decomposition (host-side, exact)
 */

#include "common.h"
#include "mont_math.cuh"

// ============================================================================
// HOST EC OPERATIONS (Montgomery domain)
// ============================================================================

static inline int h_is_infinity(const h_Point* P) {
    for (int i = 0; i < NUM_WORDS_64; i++) if (P->z.w[i] != 0) return 0;
    return 1;
}

static inline void h_set_infinity(h_Point* P) { memset(P, 0, sizeof(h_Point)); }

static inline void h_to_jacobian(const h_uint256_t* x, const h_uint256_t* y, h_Point* P) {
    P->x = *x; P->y = *y; P->z.w[0] = 1;
    for (int i = 1; i < NUM_WORDS_64; i++) P->z.w[i] = 0;
}

static inline void h_to_affine(h_AffinePoint* R, const h_Point* P) {
    if (h_is_infinity(P)) { memset(R, 0, sizeof(h_AffinePoint)); return; }
    h_uint256_t z_inv; h_modinv_mont(&P->z, &z_inv);
    h_uint256_t z_inv2, z_inv3;
    h_mont_sqr(&z_inv, &z_inv2);
    h_mont_mul(&z_inv2, &z_inv, &z_inv3);
    h_mont_mul(&P->x, &z_inv2, &R->x);
    h_mont_mul(&P->y, &z_inv3, &R->y);
}

static inline void h_batch_to_affine(h_AffinePoint* R, const h_Point* P, int count) {
    h_uint256_t* z_values = (h_uint256_t*)malloc(count * sizeof(h_uint256_t));
    h_uint256_t* z_inverses = (h_uint256_t*)malloc(count * sizeof(h_uint256_t));
    int* valid_idx = (int*)malloc(count * sizeof(int));
    int valid_count = 0;

    for (int i = 0; i < count; i++) {
        if (!h_is_infinity(&P[i])) {
            z_values[valid_count] = P[i].z;
            valid_idx[valid_count] = i;
            valid_count++;
        } else {
            memset(&R[i], 0, sizeof(h_AffinePoint));
        }
    }
    if (valid_count > 0) {
        h_batch_modinv(z_values, z_inverses, valid_count);
        for (int i = 0; i < valid_count; i++) {
            int idx = valid_idx[i];
            h_uint256_t z_inv2, z_inv3;
            h_mont_sqr(&z_inverses[i], &z_inv2);
            h_mont_mul(&z_inv2, &z_inverses[i], &z_inv3);
            h_mont_mul(&P[idx].x, &z_inv2, &R[idx].x);
            h_mont_mul(&P[idx].y, &z_inv3, &R[idx].y);
        }
    }
    free(z_values); free(z_inverses); free(valid_idx);
}

static inline void h_point_double(const h_Point* P, h_Point* R) {
    if (h_is_infinity(P)) { h_set_infinity(R); return; }
    h_uint256_t XX, YY, YYYY, S, M, T, t1, t2;
    h_mont_sqr(&P->x, &XX); h_mont_sqr(&P->y, &YY); h_mont_sqr(&YY, &YYYY);
    h_add256(&P->x, &YY, &t1); h_mont_sqr(&t1, &t1); h_sub256(&t1, &XX, &t2);
    h_sub256(&t2, &YYYY, &t1); h_add256(&t1, &t1, &S); h_add256(&XX, &XX, &t1);
    h_add256(&t1, &XX, &M); h_mont_sqr(&M, &t1); h_add256(&S, &S, &t2);
    h_sub256(&t1, &t2, &T); R->x = T; h_sub256(&S, &T, &t1); h_mont_mul(&M, &t1, &t2);
    h_add256(&YYYY, &YYYY, &t1); h_add256(&t1, &t1, &t1); h_add256(&t1, &t1, &t1);
    h_sub256(&t2, &t1, &R->y); h_mont_mul(&P->y, &P->z, &t1); h_add256(&t1, &t1, &R->z);
}

static inline void h_point_add(const h_Point* P, const h_Point* Q, h_Point* R) {
    if (h_is_infinity(P)) { *R = *Q; return; }
    if (h_is_infinity(Q)) { *R = *P; return; }

    h_uint256_t Z1Z1_check, Z2Z2_check, U1_check, U2_check, S1_check, S2_check;
    h_mont_sqr(&P->z, &Z1Z1_check); h_mont_sqr(&Q->z, &Z2Z2_check);
    h_mont_mul(&P->x, &Z2Z2_check, &U1_check); h_mont_mul(&Q->x, &Z1Z1_check, &U2_check);
    h_mont_mul(&Q->z, &Z2Z2_check, &Z2Z2_check); h_mont_mul(&P->y, &Z2Z2_check, &S1_check);
    h_mont_mul(&P->z, &Z1Z1_check, &Z1Z1_check); h_mont_mul(&Q->y, &Z1Z1_check, &S2_check);

    if (h_cmp256(&U1_check, &U2_check) == 0) {
        if (h_cmp256(&S1_check, &S2_check) == 0) {
            h_point_double(P, R);
            return;
        } else {
            h_set_infinity(R);
            return;
        }
    }
    h_uint256_t Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r, V, t1, t2;
    h_mont_sqr(&P->z, &Z1Z1); h_mont_sqr(&Q->z, &Z2Z2);
    h_mont_mul(&P->x, &Z2Z2, &U1); h_mont_mul(&Q->x, &Z1Z1, &U2);
    h_mont_mul(&Q->z, &Z2Z2, &t1); h_mont_mul(&P->y, &t1, &S1);
    h_mont_mul(&P->z, &Z1Z1, &t1); h_mont_mul(&Q->y, &t1, &S2);
    h_sub256(&U2, &U1, &H); h_mont_sqr(&H, &I); h_add256(&I, &I, &t1);
    h_add256(&t1, &I, &I); h_mont_mul(&H, &I, &J); h_sub256(&S2, &S1, &t1);
    h_add256(&t1, &t1, &r); h_mont_mul(&U1, &I, &V); h_mont_sqr(&r, &t1);
    h_sub256(&t1, &J, &t2); h_sub256(&t2, &V, &t1); h_sub256(&t1, &V, &R->x);
    h_sub256(&V, &R->x, &t1); h_mont_mul(&r, &t1, &t2); h_mont_mul(&S1, &J, &t1);
    h_add256(&t1, &t1, &t1); h_sub256(&t2, &t1, &R->y); h_add256(&P->z, &Q->z, &t1);
    h_mont_sqr(&t1, &t1); h_sub256(&t1, &Z1Z1, &t2); h_sub256(&t2, &Z2Z2, &t1);
    h_mont_mul(&t1, &H, &R->z);
}

// ============================================================================
// PIPPENGER MULTI-SCALAR MULTIPLICATION
// Computes sum(scalar[i] * point[i]) for i=0..n-1
// ============================================================================

static inline void h_pippenger(const h_uint256_t* scalars, const h_AffinePoint* points, 
                                int n, h_AffinePoint* result) {
    int num_windows = (NUM_BITS + PIPPENGER_W - 1) / PIPPENGER_W;

    h_Point total;
    h_set_infinity(&total);

    for (int w = num_windows - 1; w >= 0; w--) {
        // Double total by 2^W
        for (int b = 0; b < PIPPENGER_W; b++) {
            h_Point doubled;
            h_point_double(&total, &doubled);
            total = doubled;
        }

        // Bucket accumulation
        h_Point buckets[PIPPENGER_BUCKETS];
        for (int i = 0; i < PIPPENGER_BUCKETS; i++) h_set_infinity(&buckets[i]);

        for (int i = 0; i < n; i++) {
            int bit_pos = w * PIPPENGER_W;
            int bucket_idx = 0;
            for (int b = 0; b < PIPPENGER_W && bit_pos + b < NUM_BITS; b++) {
                int word = (bit_pos + b) >> 6, bit = (bit_pos + b) & 63;
                if ((scalars[i].w[word] >> bit) & 1ULL) bucket_idx |= (1 << b);
            }
            if (bucket_idx > 0) {
                h_Point pt;
                h_to_jacobian(&points[i].x, &points[i].y, &pt);
                h_point_add(&buckets[bucket_idx], &pt, &buckets[bucket_idx]);
            }
        }

        // Sum buckets: B[1] + 2*B[2] + 3*B[3] + ... using running sum
        h_Point running, sum;
        h_set_infinity(&running);
        h_set_infinity(&sum);

        for (int i = PIPPENGER_BUCKETS - 1; i > 0; i--) {
            h_point_add(&running, &buckets[i], &running);
            h_point_add(&sum, &running, &sum);
        }

        h_point_add(&total, &sum, &total);
    }

    h_to_affine(result, &total);
}

// ============================================================================
// WINDOWED SCALAR MULTIPLICATION (fallback)
// ============================================================================

static inline void h_build_window_table(const h_AffinePoint* G, h_AffinePoint* table, int entries) {
    memset(&table[0], 0, sizeof(h_AffinePoint));
    table[1] = *G;
    h_Point acc;
    h_to_jacobian(&G->x, &G->y, &acc);
    for (int i = 2; i < entries; i++) {
        h_Point result;
        if (i == 2) {
            h_point_double(&acc, &result);
        } else {
            h_Point g_jac;
            h_to_jacobian(&G->x, &G->y, &g_jac);
            h_point_add(&acc, &g_jac, &result);
        }
        acc = result;
        h_to_affine(&table[i], &acc);
    }
}

static inline void h_scalar_mult_windowed(const h_uint256_t* scalar, const h_AffinePoint* G, h_AffinePoint* R) {
    #define WINDOW_SIZE 4
    #define WINDOW_ENTRIES (1 << WINDOW_SIZE)

    h_AffinePoint table[WINDOW_ENTRIES];
    h_build_window_table(G, table, WINDOW_ENTRIES);

    h_Point result;
    h_set_infinity(&result);
    int total_bits = NUM_BITS;
    int num_windows = (total_bits + WINDOW_SIZE - 1) / WINDOW_SIZE;

    for (int w = num_windows - 1; w >= 0; w--) {
        int bit_pos = w * WINDOW_SIZE;
        int window_val = 0;
        for (int b = 0; b < WINDOW_SIZE; b++) {
            int bit_idx = bit_pos + b;
            if (bit_idx < total_bits) {
                int word = bit_idx >> 6, bit = bit_idx & 63;
                if ((scalar->w[word] >> bit) & 1ULL) window_val |= (1 << b);
            }
        }
        if (w < num_windows - 1) {
            for (int b = 0; b < WINDOW_SIZE; b++) {
                h_Point doubled;
                h_point_double(&result, &doubled);
                result = doubled;
            }
        }
        if (window_val > 0) {
            h_Point addend;
            h_to_jacobian(&table[window_val].x, &table[window_val].y, &addend);
            h_Point tmp;
            h_point_add(&result, &addend, &tmp);
            result = tmp;
        }
    }
    h_to_affine(R, &result);
}
