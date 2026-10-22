/**
 * ec_ops.cuh — Elliptic curve operations
 * - SoA (Structure of Arrays) layout
 * - Montgomery domain arithmetic
 * - Mixed addition (Z=1 fast path)
 * - Point doubling (optimized)
 * - Batch inversion via Montgomery trick
 */

#ifndef PSC_EC_OPS_CUH
#define PSC_EC_OPS_CUH

#include "common.h"
#include "mont_math.cuh"

// ============================================================================
// SoA HELPERS
// ============================================================================

__device__ __forceinline__ uint64_t* soa_x_ptr(KangarooSoA* soa, int k) {
    return &soa->x[k * NUM_WORDS_64];
}

__device__ __forceinline__ uint64_t* soa_y_ptr(KangarooSoA* soa, int k) {
    return &soa->y[k * NUM_WORDS_64];
}

__device__ __forceinline__ uint64_t* soa_z_ptr(KangarooSoA* soa, int k) {
    return &soa->z[k * NUM_WORDS_64];
}

__device__ __forceinline__ uint64_t* soa_dist_ptr(KangarooSoA* soa, int k) {
    return &soa->dist[k * NUM_WORDS_64];
}

__device__ __forceinline__ void soa_load_x(const KangarooSoA* soa, int k, uint256_t* x) {
    const uint64_t* p = &soa->x[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) x->w[i] = p[i];
}

__device__ __forceinline__ void soa_load_y(const KangarooSoA* soa, int k, uint256_t* y) {
    const uint64_t* p = &soa->y[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) y->w[i] = p[i];
}

__device__ __forceinline__ void soa_load_z(const KangarooSoA* soa, int k, uint256_t* z) {
    const uint64_t* p = &soa->z[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) z->w[i] = p[i];
}

__device__ __forceinline__ void soa_load_dist(const KangarooSoA* soa, int k, uint256_t* dist) {
    const uint64_t* p = &soa->dist[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) dist->w[i] = p[i];
}

__device__ __forceinline__ void soa_store_x(KangarooSoA* soa, int k, const uint256_t* x) {
    uint64_t* p = &soa->x[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) p[i] = x->w[i];
}

__device__ __forceinline__ void soa_store_y(KangarooSoA* soa, int k, const uint256_t* y) {
    uint64_t* p = &soa->y[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) p[i] = y->w[i];
}

__device__ __forceinline__ void soa_store_z(KangarooSoA* soa, int k, const uint256_t* z) {
    uint64_t* p = &soa->z[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) p[i] = z->w[i];
}

__device__ __forceinline__ void soa_store_dist(KangarooSoA* soa, int k, const uint256_t* dist) {
    uint64_t* p = &soa->dist[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) p[i] = dist->w[i];
}

__device__ __forceinline__ int soa_is_infinity(const KangarooSoA* soa, int k) {
    const uint64_t* p = &soa->z[k * NUM_WORDS_64];
    return (p[0] | p[1] | p[2] | p[3]) == 0;
}

__device__ __forceinline__ void soa_set_infinity(KangarooSoA* soa, int k) {
    uint64_t* px = &soa->x[k * NUM_WORDS_64];
    uint64_t* py = &soa->y[k * NUM_WORDS_64];
    uint64_t* pz = &soa->z[k * NUM_WORDS_64];
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        px[i] = 0; py[i] = 0; pz[i] = 0;
    }
}

__device__ __forceinline__ void soa_to_jacobian(KangarooSoA* soa, int k, const uint256_t* x, const uint256_t* y) {
    soa_store_x(soa, k, x);
    soa_store_y(soa, k, y);
    uint64_t* pz = &soa->z[k * NUM_WORDS_64];
    pz[0] = 1; pz[1] = 0; pz[2] = 0; pz[3] = 0;
}

__device__ __forceinline__ int soa_is_z1(const KangarooSoA* soa, int k) {
    const uint64_t* p = &soa->z[k * NUM_WORDS_64];
    return (p[0] == 1) & (p[1] == 0) & (p[2] == 0) & (p[3] == 0);
}

// ============================================================================
// POINT DOUBLING (Montgomery, optimized, 6 temporaries)
// ============================================================================

__device__ __forceinline__ void point_double_mont_soa(const KangarooSoA* in, int k, KangarooSoA* out) {
    if (soa_is_infinity(in, k)) { soa_set_infinity(out, k); return; }

    uint256_t X, Y, Z;
    soa_load_x(in, k, &X);
    soa_load_y(in, k, &Y);
    soa_load_z(in, k, &Z);

    uint256_t XX, YY, M, S, T, t1, t2;

    mont_sqr(&X, &XX);           // XX = X^2
    mont_sqr(&Y, &YY);           // YY = Y^2

    add256_v2(&XX, &XX, &t1);    // t1 = 2*XX
    add256_v2(&t1, &XX, &M);     // M = 3*XX

    mont_mul(&X, &YY, &t1);      // t1 = X*YY
    add256_v2(&t1, &t1, &S);     // S = 2*X*YY
    add256_v2(&S, &S, &S);       // S = 4*X*YY (actually 2*t1 = 4*X*YY)

    mont_sqr(&M, &t1);           // t1 = M^2
    sub256(&t1, &S, &T);         // T = M^2 - S
    sub256(&T, &S, &T);          // T = M^2 - 2*S
    soa_store_x(out, k, &T);     // X3 = T

    sub256(&S, &T, &t1);         // t1 = S - T
    mont_mul(&M, &t1, &t2);      // t2 = M*(S-T)

    mont_sqr(&YY, &t1);          // t1 = YY^2
    add256_v2(&t1, &t1, &YY);    // YY = 2*YY^2
    add256_v2(&YY, &YY, &YY);    // YY = 4*YY^2
    add256_v2(&YY, &YY, &YY);    // YY = 8*YY^2 (wait, that's wrong)
    // Correct: 2*(YY^2 + YY^2) = 4*YY^2, then double = 8*YY^2? No.
    // Actually: YYYY = YY^2, then 2*YYYY = 2*YY^2
    // Let me recalculate: we need 8*YYYY where YYYY = YY^2
    // t1 = YY^2, then t1 + t1 = 2*t1, + t1 = 3*t1... no

    // Simplified: just compute 8*YY^2 directly
    // Actually, original formula is: 8*YYYY where YYYY = YY^2
    // So: YYYY = YY^2, then 8*YYYY = YYYY * 8

    // Let me fix: compute YYYY = YY^2, then 8*YYYY
    // t1 already = YY^2 from mont_sqr above
    // But we overwrote t1... Let me restructure

    // Actually, looking at the original formula:
    // Y3 = M*(S-T) - 8*YYYY where YYYY = YY^2
    // We have t2 = M*(S-T), need 8*YYYY

    // Recompute YY^2 into t1
    mont_sqr(&YY, &t1);          // t1 = YY^2 = YYYY
    add256_v2(&t1, &t1, &t1);    // t1 = 2*YYYY
    add256_v2(&t1, &t1, &t1);    // t1 = 4*YYYY
    add256_v2(&t1, &t1, &t1);    // t1 = 8*YYYY

    sub256(&t2, &t1, &t1);       // t1 = M*(S-T) - 8*YYYY
    soa_store_y(out, k, &t1);    // Y3 = t1

    mont_mul(&Y, &Z, &t1);       // t1 = Y*Z
    add256_v2(&t1, &t1, &t2);    // t2 = 2*Y*Z
    soa_store_z(out, k, &t2);    // Z3 = 2*Y*Z
}

// ============================================================================
// MIXED ADDITION (Z=1 fast path) — Montgomery
// ============================================================================

__device__ __forceinline__ void point_add_mixed_Z1_mont_soa(const KangarooSoA* P, int pk, 
                                                             const AffinePoint* Q, 
                                                             KangarooSoA* R, int rk) {
    uint256_t PX, PY;
    soa_load_x(P, pk, &PX);
    soa_load_y(P, pk, &PY);

    uint256_t H, HH, I, J, r, V, t1, t2, RX, RY;

    sub256((const uint256_t*)&Q->x, &PX, &H);   // H = Qx - Px
    mont_sqr(&H, &HH);                           // HH = H^2
    add256_v2(&HH, &HH, &I);                     // I = 2*HH
    add256_v2(&I, &I, &I);                       // I = 4*HH
    mont_mul(&H, &I, &J);                        // J = H*I

    sub256((const uint256_t*)&Q->y, &PY, &t1);  // t1 = Qy - Py
    add256_v2(&t1, &t1, &r);                     // r = 2*(Qy - Py)

    mont_mul(&PX, &I, &V);                       // V = Px*I

    mont_sqr(&r, &t1);                           // t1 = r^2
    sub256(&t1, &J, &t2);                        // t2 = r^2 - J
    sub256(&t2, &V, &t1);                        // t1 = r^2 - J - V
    sub256(&t1, &V, &RX);                        // Rx = r^2 - J - 2*V

    sub256(&V, &RX, &t1);                        // t1 = V - Rx
    mont_mul(&r, &t1, &t2);                      // t2 = r*(V-Rx)

    mont_mul(&PY, &J, &t1);                      // t1 = Py*J
    add256_v2(&t1, &t1, &t1);                    // t1 = 2*Py*J
    sub256(&t2, &t1, &RY);                       // Ry = r*(V-Rx) - 2*Py*J

    soa_store_x(R, rk, &RX);
    soa_store_y(R, rk, &RY);
    soa_store_z(R, rk, &H);                      // Z = H
}

// ============================================================================
// MIXED ADDITION (general Z) — Montgomery
// ============================================================================

__device__ __forceinline__ void point_add_mixed_mont_soa(const KangarooSoA* P, int pk,
                                                          const AffinePoint* Q,
                                                          KangarooSoA* R, int rk) {
    if (soa_is_infinity(P, pk)) {
        soa_to_jacobian(R, rk, &Q->x, &Q->y);
        return;
    }

    if (soa_is_z1(P, pk)) {
        point_add_mixed_Z1_mont_soa(P, pk, Q, R, rk);
        return;
    }

    uint256_t PX, PY, PZ;
    soa_load_x(P, pk, &PX);
    soa_load_y(P, pk, &PY);
    soa_load_z(P, pk, &PZ);

    uint256_t Z1Z1, U2, S2, H, HH, I, J, r, V, t1, t2, RX, RY, RZ;

    mont_sqr(&PZ, &Z1Z1);                        // Z1Z1 = Z^2
    mont_mul(&Q->x, &Z1Z1, &U2);                 // U2 = Qx * Z1Z1
    mont_mul(&PZ, &Z1Z1, &t1);                   // t1 = Z * Z1Z1
    mont_mul(&Q->y, &t1, &S2);                   // S2 = Qy * Z^3

    sub256(&U2, &PX, &H);                        // H = U2 - Px
    mont_sqr(&H, &HH);                           // HH = H^2
    add256_v2(&HH, &HH, &I);                     // I = 2*HH
    add256_v2(&I, &I, &I);                       // I = 4*HH
    mont_mul(&H, &I, &J);                        // J = H*I

    sub256(&S2, &PY, &t1);                       // t1 = S2 - Py
    add256_v2(&t1, &t1, &r);                     // r = 2*(S2 - Py)

    mont_mul(&PX, &I, &V);                       // V = Px*I

    mont_sqr(&r, &t1);                           // t1 = r^2
    sub256(&t1, &J, &t2);                        // t2 = r^2 - J
    sub256(&t2, &V, &t1);                        // t1 = r^2 - J - V
    sub256(&t1, &V, &RX);                        // Rx = r^2 - J - 2*V

    sub256(&V, &RX, &t1);                        // t1 = V - Rx
    mont_mul(&r, &t1, &t2);                      // t2 = r*(V-Rx)

    mont_mul(&PY, &J, &t1);                      // t1 = Py*J
    add256_v2(&t1, &t1, &t1);                    // t1 = 2*Py*J
    sub256(&t2, &t1, &RY);                       // Ry = r*(V-Rx) - 2*Py*J

    add256_v2(&PZ, &H, &t1);                     // t1 = Z + H
    mont_sqr(&t1, &t1);                          // t1 = (Z+H)^2
    sub256(&t1, &Z1Z1, &t2);                     // t2 = (Z+H)^2 - Z1Z1
    sub256(&t2, &HH, &t1);                       // t1 = (Z+H)^2 - Z1Z1 - HH
    mont_mul(&t1, &H, &RZ);                      // Rz = ((Z+H)^2 - Z1Z1 - HH) * H

    soa_store_x(R, rk, &RX);
    soa_store_y(R, rk, &RY);
    soa_store_z(R, rk, &RZ);
}

// ============================================================================
// BATCH INVERSION — Montgomery trick on GPU (shared memory)
// Inverts up to 8 values per warp using parallel prefix products
// ============================================================================

__device__ __forceinline__ void warp_batch_invert_mont(uint256_t* values, uint256_t* inverses, int count) {
    if (count == 0) return;
    if (count == 1) { modinv_mont(&values[0], &inverses[0]); return; }

    // Parallel prefix product within warp
    // Each lane computes partial product, then broadcast
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane = warp.thread_rank();

    // Simple sequential version for now (can be parallelized)
    uint256_t products[8];
    products[0] = values[0];
    #pragma unroll
    for (int i = 1; i < count; i++) {
        mont_mul(&products[i-1], &values[i], &products[i]);
    }

    uint256_t inverse;
    modinv_mont(&products[count-1], &inverse);

    #pragma unroll
    for (int i = count - 1; i > 0; i--) {
        mont_mul(&inverse, &products[i-1], &inverses[i]);
        mont_mul(&inverse, &values[i], &inverse);
    }
    inverses[0] = inverse;
}

// ============================================================================
// NEGATION MAP — Canonicalize Y (Montgomery domain)
// ============================================================================

__device__ __forceinline__ void soa_canonicalize_y_mont(KangarooSoA* soa, int k) {
    uint256_t y;
    soa_load_y(soa, k, &y);
    int need_flip = cmp256(&y, (const uint256_t*)d_p_half) > 0;
    uint256_t flipped;
    sub256((const uint256_t*)d_p, &y, &flipped);
    #pragma unroll
    for (int i = 0; i < NUM_WORDS_64; i++) {
        y.w[i] = need_flip ? flipped.w[i] : y.w[i];
    }
    soa_store_y(soa, k, &y);
}

#endif // PSC_EC_OPS_CUH
