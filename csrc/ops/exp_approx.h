/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Fast exp approximation utilities (6th-order Horner + range reduction).
 * Common header for all attention backends.
 *
 * Usage:
 *   // Hoist constants before the loop:
 *   EXP_APPROX_AVX512_CONSTANTS();
 *
 *   for (...) {
 *     __m512 result;
 *     EXP_APPROX_AVX512(input_vec, result);
 *     float scalar_result;
 *     EXP_APPROX_SCALAR(input_val, scalar_result);
 *   }
 *
 * REQUIRES: -mavx512f -mavx512bf16 compiler flags.
 ******************************************************************************/

#ifndef PACE_OPS_EXP_APPROX_H
#define PACE_OPS_EXP_APPROX_H

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

// AVX-512 vectorized exp approximation (16 floats, ~1e-6 relative accuracy)
// Declare constants — call once before the loop that uses EXP_APPROX_AVX512.
// These are local variables that the macro body references by name.
// clang-format off
#define EXP_APPROX_AVX512_CONSTANTS()                                           \
  __attribute__((unused)) const __m512 _ea_min   = _mm512_set1_ps(-88.0f);     \
  __attribute__((unused)) const __m512 _ea_log2e = _mm512_set1_ps(1.442695041f); \
  __attribute__((unused)) const __m512 _ea_ln2hi = _mm512_set1_ps(0.693145752f); \
  __attribute__((unused)) const __m512 _ea_ln2lo = _mm512_set1_ps(1.42860677e-6f); \
  __attribute__((unused)) const __m512 _ea_one   = _mm512_set1_ps(1.0f);       \
  __attribute__((unused)) const __m512 _ea_c5    = _mm512_set1_ps(0.001388889f); \
  __attribute__((unused)) const __m512 _ea_c4    = _mm512_set1_ps(0.008333333f); \
  __attribute__((unused)) const __m512 _ea_c3    = _mm512_set1_ps(0.041666667f); \
  __attribute__((unused)) const __m512 _ea_c2    = _mm512_set1_ps(0.166666667f); \
  __attribute__((unused)) const __m512 _ea_c1    = _mm512_set1_ps(0.5f);       \
  __attribute__((unused)) const __m512i _ea_i127 = _mm512_set1_epi32(127)

// Compute exp(x) for 16 floats. Uses constants from EXP_APPROX_AVX512_CONSTANTS.
#define EXP_APPROX_AVX512(x, result) do {                               \
  __m512 _ea_x = _mm512_max_ps((x), _ea_min);                          \
  __m512 _ea_n = _mm512_roundscale_ps(                                  \
      _mm512_mul_ps(_ea_x, _ea_log2e), _MM_FROUND_TO_NEAREST_INT);     \
  __m512 _ea_r = _mm512_fnmadd_ps(_ea_n, _ea_ln2hi, _ea_x);            \
  _ea_r = _mm512_fnmadd_ps(_ea_n, _ea_ln2lo, _ea_r);                   \
  __m512 _ea_p = _mm512_fmadd_ps(_ea_r, _ea_c5, _ea_c4);               \
  _ea_p = _mm512_fmadd_ps(_ea_r, _ea_p, _ea_c3);                       \
  _ea_p = _mm512_fmadd_ps(_ea_r, _ea_p, _ea_c2);                       \
  _ea_p = _mm512_fmadd_ps(_ea_r, _ea_p, _ea_c1);                       \
  _ea_p = _mm512_fmadd_ps(_ea_r, _ea_p, _ea_one);                      \
  _ea_p = _mm512_fmadd_ps(_ea_r, _ea_p, _ea_one);                      \
  __m512i _ea_ni = _mm512_cvtps_epi32(_ea_n);                           \
  _ea_ni = _mm512_add_epi32(_ea_ni, _ea_i127);                          \
  _ea_ni = _mm512_slli_epi32(_ea_ni, 23);                               \
  (result) = _mm512_mul_ps(_ea_p, _mm512_castsi512_ps(_ea_ni));         \
} while (0)
// clang-format on

// Scalar exp approximation (same polynomial, single float).
// No constants macro needed — scalar uses literal constants inline.
// clang-format off
#define EXP_APPROX_SCALAR(x, result) do {                               \
  float _es_x = (x);                                                    \
  if (_es_x < -88.0f) { (result) = 0.0f; break; }                      \
  float _es_n = std::nearbyint(_es_x * 1.442695041f);                   \
  float _es_r = _es_x - _es_n * 0.693145752f - _es_n * 1.42860677e-6f; \
  float _es_p = 0.001388889f;                                           \
  _es_p = _es_r * _es_p + 0.008333333f;                                 \
  _es_p = _es_r * _es_p + 0.041666667f;                                 \
  _es_p = _es_r * _es_p + 0.166666667f;                                 \
  _es_p = _es_r * _es_p + 0.5f;                                         \
  _es_p = _es_r * _es_p + 1.0f;                                         \
  _es_p = _es_r * _es_p + 1.0f;                                         \
  uint32_t _es_ni =                                                      \
      static_cast<uint32_t>(static_cast<int32_t>(_es_n) + 127);          \
  _es_ni <<= 23;                                                        \
  float _es_scale;                                                       \
  std::memcpy(&_es_scale, &_es_ni, sizeof(float));                       \
  (result) = _es_p * _es_scale;                                          \
} while (0)
// clang-format on

#endif // PACE_OPS_EXP_APPROX_H
