// lej_bench.c
// Single-thread mul+load heavy benchmark for dynamic analysis / dependency-chain tools.
// Build: gcc -O2 -g -std=c11 -Wall -Wextra -march=native lej_bench.c -o lej_bench
// Run:   ./lej_bench
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#if defined(__linux__)
  #include <unistd.h>
#endif

// ---------------- timing ----------------
static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---------------- deterministic rng ----------------
static inline uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// ---------------- aligned alloc ----------------
static void* xaligned_alloc(size_t align, size_t size) {
#if (__STDC_VERSION__ >= 201112L)
  // aligned_alloc requires size multiple of align
  size_t padded = (size + align - 1) & ~(align - 1);
  void* p = aligned_alloc(align, padded);
  if (!p) { perror("aligned_alloc"); exit(1); }
  return p;
#else
  void* p = NULL;
  if (posix_memalign(&p, align, size) != 0 || !p) { perror("posix_memalign"); exit(1); }
  return p;
#endif
}

// ---------------- checksum to keep optimizer honest ----------------
static uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static uint64_t checksum_f32(const float* a, size_t n) {
  uint64_t h = 0x123456789abcdef0ULL;
  for (size_t i = 0; i < n; i += 97) { // sample stride to reduce overhead
    uint32_t u;
    memcpy(&u, &a[i], sizeof(u));
    h = mix64(h ^ (uint64_t)u ^ (uint64_t)i);
  }
  return h;
}

// ---------------- kernel 1: blocked GEMM ----------------
// C = A*B + C (accumulate). Mul-heavy.
// Matrices are row-major: A[NxN], B[NxN], C[NxN]
static void gemm_blocked(float* C, const float* A, const float* B, int N, int BS, int reps) {
  for (int r = 0; r < reps; r++) {
    for (int ii = 0; ii < N; ii += BS) {
      for (int kk = 0; kk < N; kk += BS) {
        for (int jj = 0; jj < N; jj += BS) {
          int i_max = (ii + BS < N) ? (ii + BS) : N;
          int k_max = (kk + BS < N) ? (kk + BS) : N;
          int j_max = (jj + BS < N) ? (jj + BS) : N;

          for (int i = ii; i < i_max; i++) {
            for (int k = kk; k < k_max; k++) {
              float aik = A[i * N + k];
              // inner loop is mul-add heavy
              for (int j = jj; j < j_max; j++) {
                C[i * N + j] += aik * B[k * N + j];
              }
            }
          }
        }
      }
    }
  }
}

// ---------------- kernel 2: gather + FMA streaming ----------------
// y[i] = y[i] + x[idx[i]] * w[i] (load-heavy due to idx indirection; still mul-heavy).
static void gather_fma(float* y, const float* x, const float* w, const uint32_t* idx,
                       size_t n, int reps) {
  for (int r = 0; r < reps; r++) {
    // Unrolled loop to create multiple independent accumulations (more interesting mix)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      float a0 = x[idx[i+0]] * w[i+0];
      float a1 = x[idx[i+1]] * w[i+1];
      float a2 = x[idx[i+2]] * w[i+2];
      float a3 = x[idx[i+3]] * w[i+3];
      float a4 = x[idx[i+4]] * w[i+4];
      float a5 = x[idx[i+5]] * w[i+5];
      float a6 = x[idx[i+6]] * w[i+6];
      float a7 = x[idx[i+7]] * w[i+7];

      y[i+0] += a0; y[i+1] += a1; y[i+2] += a2; y[i+3] += a3;
      y[i+4] += a4; y[i+5] += a5; y[i+6] += a6; y[i+7] += a7;
    }
    for (; i < n; i++) {
      y[i] += x[idx[i]] * w[i];
    }
  }
}

// ---------------- kernel 3: pointer chase + dependent MAC ----------------
// This produces a *strong* dependency chain:
//   p = next[p]  (dependent load)
//   acc = acc * alpha + data[p] * beta (mul+load dependency)
static float pointer_chase_mac(const uint32_t* next, const float* data, uint32_t start,
                               size_t steps, float alpha, float beta) {
  uint32_t p = start;
  float acc = 1.0f;

  // Make the chain long & hard for out-of-order: each step depends on previous p and acc.
  for (size_t i = 0; i < steps; i++) {
    p = next[p];
    float v = data[p];
    acc = acc * alpha + v * beta;
    // small nonlinear update prevents easy algebraic simplification
    acc = acc + 1e-6f * (float)(p & 1023u);
  }
  return acc;
}

int main(void) {
  // --------- sizes (tune these if you want heavier/lighter) ----------
  const int N = 512;          // GEMM matrix dimension: 512 => ~134M mul-adds per rep
  const int BS = 32;          // block size
  const int GEMM_REPS = 2;    // increase for heavier compute

  const size_t XN = 16u * 1024u * 1024u; // 16M floats (~64MB) for gather source (load-heavy)
  const size_t GN = 8u  * 1024u * 1024u; // 8M outputs (~32MB)
  const int GATHER_REPS = 3;

  const uint32_t PNN = 8u * 1024u * 1024u; // 8M nodes for pointer chase (~32MB next + ~32MB data)
  const size_t P_STEPS = 50u * 1000u * 1000u; // 50M steps (dependency chain)
  const float ALPHA = 1.000001f;
  const float BETA  = 0.999999f;

  // --------- allocate aligned ----------
  float* A = (float*)xaligned_alloc(64, (size_t)N * (size_t)N * sizeof(float));
  float* B = (float*)xaligned_alloc(64, (size_t)N * (size_t)N * sizeof(float));
  float* C = (float*)xaligned_alloc(64, (size_t)N * (size_t)N * sizeof(float));

  float* x = (float*)xaligned_alloc(64, XN * sizeof(float));
  float* w = (float*)xaligned_alloc(64, GN * sizeof(float));
  float* y = (float*)xaligned_alloc(64, GN * sizeof(float));
  uint32_t* idx = (uint32_t*)xaligned_alloc(64, GN * sizeof(uint32_t));

  uint32_t* next = (uint32_t*)xaligned_alloc(64, (size_t)PNN * sizeof(uint32_t));
  float* pdata = (float*)xaligned_alloc(64, (size_t)PNN * sizeof(float));

  // --------- init data deterministically ----------
  uint64_t seed = 0xC0FFEE123456789ull;

  for (int i = 0; i < N * N; i++) {
    uint64_t r = splitmix64(&seed);
    A[i] = (float)((int)(r & 2047) - 1024) / 1024.0f;
    B[i] = (float)((int)((r >> 11) & 2047) - 1024) / 1024.0f;
    C[i] = 0.0f;
  }

  for (size_t i = 0; i < XN; i++) {
    uint64_t r = splitmix64(&seed);
    x[i] = (float)((int)(r & 65535) - 32768) / 32768.0f;
  }
  for (size_t i = 0; i < GN; i++) {
    uint64_t r = splitmix64(&seed);
    w[i] = (float)((int)(r & 8191) - 4096) / 4096.0f;
    y[i] = 0.1f * (float)(i & 1023u);

    // Random but bounded indices => load-heavy and poor locality
    idx[i] = (uint32_t)(splitmix64(&seed) % XN);
  }

  // Build a single-cycle permutation for pointer chasing (max dependency / no early termination)
  // Fisher-Yates shuffle to make next[] a permutation; then link as cycle.
  for (uint32_t i = 0; i < PNN; i++) next[i] = i;
  for (uint32_t i = PNN - 1; i > 0; i--) {
    uint32_t j = (uint32_t)(splitmix64(&seed) % (uint64_t)(i + 1));
    uint32_t tmp = next[i];
    next[i] = next[j];
    next[j] = tmp;
  }
  // Convert permutation array into next pointers forming one big cycle:
  // node perm[k] -> perm[k+1], last -> first
  // We currently have next[] holding perm list; reuse a temp pointer mapping.
  uint32_t* perm = next; // alias
  uint32_t* nextptr = (uint32_t*)xaligned_alloc(64, (size_t)PNN * sizeof(uint32_t));
  for (uint32_t k = 0; k + 1 < PNN; k++) nextptr[perm[k]] = perm[k + 1];
  nextptr[perm[PNN - 1]] = perm[0];
  // fill pdata
  for (uint32_t i = 0; i < PNN; i++) {
    uint64_t r = splitmix64(&seed);
    pdata[i] = (float)((int)(r & 131071) - 65536) / 65536.0f;
  }
  // swap into next
  memcpy(next, nextptr, (size_t)PNN * sizeof(uint32_t));
  free(nextptr);

  // --------- run benchmark ----------
  uint64_t t0 = now_ns();

  uint64_t t_gemm0 = now_ns();
  gemm_blocked(C, A, B, N, BS, GEMM_REPS);
  uint64_t t_gemm1 = now_ns();

  uint64_t t_gath0 = now_ns();
  gather_fma(y, x, w, idx, GN, GATHER_REPS);
  uint64_t t_gath1 = now_ns();

  uint64_t t_ptr0 = now_ns();
  uint32_t start = (uint32_t)(splitmix64(&seed) % PNN);
  float acc = pointer_chase_mac(next, pdata, start, P_STEPS, ALPHA, BETA);
  uint64_t t_ptr1 = now_ns();

  uint64_t t1 = now_ns();

  // --------- checksums (avoid DCE) ----------
  uint64_t cC = checksum_f32(C, (size_t)N * (size_t)N);
  uint64_t cY = checksum_f32(y, GN);
  uint64_t cP;
  {
    uint32_t u; memcpy(&u, &acc, sizeof(u));
    cP = mix64((uint64_t)u);
  }
  uint64_t final = mix64(cC ^ (cY + 0x9e3779b97f4a7c15ULL) ^ (cP << 1));

  // --------- report ----------
  double s_total = (double)(t1 - t0) / 1e9;
  double s_gemm  = (double)(t_gemm1 - t_gemm0) / 1e9;
  double s_gath  = (double)(t_gath1 - t_gath0) / 1e9;
  double s_ptr   = (double)(t_ptr1  - t_ptr0) / 1e9;

  printf("lej_bench: N=%d BS=%d GEMM_REPS=%d | XN=%zu GN=%zu GATHER_REPS=%d | PNN=%u P_STEPS=%zu\n",
         N, BS, GEMM_REPS, XN, GN, GATHER_REPS, PNN, P_STEPS);
  printf("time: total=%.3f s | gemm=%.3f s | gather=%.3f s | ptrchase=%.3f s\n",
         s_total, s_gemm, s_gath, s_ptr);
  printf("checks: C=0x%016llx Y=0x%016llx acc=%g final=0x%016llx\n",
         (unsigned long long)cC, (unsigned long long)cY, (double)acc, (unsigned long long)final);

  // cleanup
  free(A); free(B); free(C);
  free(x); free(w); free(y); free(idx);
  free(next); free(pdata);

  // Return value depends on final to prevent whole-program DCE in some LTO scenarios
  return (int)(final & 0x7fffffffULL);
}