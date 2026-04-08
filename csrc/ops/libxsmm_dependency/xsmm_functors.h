/**********************************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 **********************************************************************************************/

/**********************************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/
 * Source Code:
 * https://github.com/libxsmm/tpp-pytorch-extension/blob/mlperf_infer_31/src/csrc/xsmm_functors.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 **********************************************************************************************/

/* Author: Dhiraj Kalamkar (Intel Corp.)
 **********************************************************************************************/

#ifndef _XSMM_FUNCTORS_H_
#define _XSMM_FUNCTORS_H_

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#else
// #include "pytorch_extension_wrapper.h"
#endif
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <string>
#include <unordered_map>

#define TPP_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)
#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
// extern long long hsh_key, hsh_ret;
namespace tpp {
typedef at::BFloat16 bfloat16;
typedef at::Half half;
// typedef at::BFloat8 bfloat8;
inline float upconvert_to_float(float val) {
  return val;
}
inline float upconvert_to_float(bfloat16 val) {
  return (float)val;
}
inline float upconvert_to_float(half val) {
  return (float)val;
}

template <typename T>
inline libxsmm_datatype XsmmDtype();
template <>
inline libxsmm_datatype XsmmDtype<int64_t>() {
  return LIBXSMM_DATATYPE_I64;
}
template <>
inline libxsmm_datatype XsmmDtype<int32_t>() {
  return LIBXSMM_DATATYPE_I32;
}
template <>
inline libxsmm_datatype XsmmDtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}
template <>
inline libxsmm_datatype XsmmDtype<bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}
template <>
inline libxsmm_datatype XsmmDtype<half>() {
  return LIBXSMM_DATATYPE_F16;
}

#ifdef __AVX512F__
inline __m512 _mm512_loadu_ps_auto(float const* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, float const* mem_addr) {
  return _mm512_maskz_loadu_ps(k, mem_addr);
}
inline void _mm512_storeu_ps_auto(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}
inline void _mm512_mask_storeu_ps_auto(float* mem_addr, __mmask16 k, __m512 a) {
  _mm512_mask_storeu_ps(mem_addr, k, a);
}

inline __m512 _mm512_loadu_ps_auto(half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, half const* mem_addr) {
  return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(half* mem_addr, __m512 a) {
  _mm256_storeu_si256(
      (__m256i*)mem_addr,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
inline void _mm512_mask_storeu_ps_auto(half* mem_addr, __mmask16 k, __m512 a) {
  _mm256_mask_storeu_epi16(
      (__m256i*)mem_addr,
      k,
      _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m512 _mm512_convert_bf_ps(__m256i a) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a), 16));
}
inline __m256i _mm256_convert_ps_bf(__m512 a) {
  return _mm512_cvtepi32_epi16(
      _mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a), 16));
}

inline __m512 _mm512_loadu_ps_auto(bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));
}
inline __m512 _mm512_maskz_loadu_ps_auto(
    __mmask16 k,
    bfloat16 const* mem_addr) {
  return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));
}
inline void _mm512_storeu_ps_auto(bfloat16* mem_addr, __m512 a) {
  _mm256_storeu_si256((__m256i*)mem_addr, _mm256_convert_ps_bf(a));
}
inline void _mm512_mask_storeu_ps_auto(
    bfloat16* mem_addr,
    __mmask16 k,
    __m512 a) {
  _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a));
}

inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline __m512 _mm512_maskz_split_loadu_ps(
    __mmask16 k,
    bfloat16 const* hi,
    bfloat16 const* lo) {
  auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
  auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
  return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
}
inline void _mm512_split_storeu_ps(bfloat16* hi, bfloat16* lo, __m512 a) {
  //_mm512_storeu_ps_auto(hi, a);
  _mm256_storeu_si256(
      (__m256i*)hi,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_storeu_si256(
      (__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
inline void _mm512_mask_split_storeu_ps(
    bfloat16* hi,
    bfloat16* lo,
    __mmask16 k,
    __m512 a) {
  //_mm512_mask_storeu_ps_auto(hi, k, a);
  _mm256_mask_storeu_epi16(
      (__m256i*)hi,
      k,
      _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
  _mm256_mask_storeu_epi16(
      (__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
}
}
#endif

inline libxsmm_datatype convert_dtype_pt2xsmm(at::ScalarType dtype) {
  static const std::map<at::ScalarType, libxsmm_datatype> pt2xsmmDtypes = {
      {at::kDouble, LIBXSMM_DATATYPE_F64},
      {at::kFloat, LIBXSMM_DATATYPE_F32},
      {at::kHalf, LIBXSMM_DATATYPE_F16},
      {at::kBFloat16, LIBXSMM_DATATYPE_BF16},
      {at::kByte, LIBXSMM_DATATYPE_I8},
      {at::kChar, LIBXSMM_DATATYPE_I8},
      {at::kShort, LIBXSMM_DATATYPE_I16},
      {at::kInt, LIBXSMM_DATATYPE_I32},
      {at::kLong, LIBXSMM_DATATYPE_I64}};

  return pt2xsmmDtypes.at(dtype);
}

inline int xsmm_get_vnni_block_size(libxsmm_datatype dtype) {
  int bs = libxsmm_cpuid_dot_pack_factor(dtype);
  if (bs <= 0) {
    throw std::invalid_argument("Unsupported datatype");
  }
  return bs;
}

inline int get_vnni_block_size(at::ScalarType dtype) {
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline int get_vnni_block_size(caffe2::TypeMeta dtype_) {
  at::ScalarType dtype = dtype_.toScalarType();
  auto xsmm_dtype = convert_dtype_pt2xsmm(dtype);
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

template <typename T>
inline int get_vnni_block_size() {
  auto xsmm_dtype = XsmmDtype<T>();
  return xsmm_get_vnni_block_size(xsmm_dtype);
}

inline void debug_print_eqn_tree(libxsmm_blasint eqn_no) {
  if (false) {
    libxsmm_meqn_tree_print(eqn_no);
    libxsmm_meqn_rpn_print(eqn_no);
  }
}

inline int meqn_push_arg(
    const libxsmm_blasint idx,
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint ld,
    const libxsmm_blasint in_pos,
    const libxsmm_blasint offs_in_pos,
    const libxsmm_datatype dtype) {
  // This "singular" type dictates that the arg is a regular tensor (and not a
  // set of tensors)
  libxsmm_matrix_arg_attributes arg_singular_attr =
      libxsmm_create_matrix_arg_attributes(
          LIBXSMM_MATRIX_ARG_TYPE_SINGULAR,
          LIBXSMM_MATRIX_ARG_SET_TYPE_NONE,
          0,
          0);
  // Arg metadata include equation id and pos in arg array at runtime
  libxsmm_meqn_arg_metadata arg_metadata =
      libxsmm_create_meqn_arg_metadata(idx, in_pos);
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, ld, dtype);
  return libxsmm_meqn_push_back_arg(arg_metadata, arg_shape, arg_singular_attr);
}

inline libxsmm_meqn_function meqn_dispatch(
    const libxsmm_blasint m,
    const libxsmm_blasint n,
    const libxsmm_blasint* ldo,
    const libxsmm_datatype out_type,
    const unsigned int idx) {
  libxsmm_meqn_arg_shape arg_shape =
      libxsmm_create_meqn_arg_shape(m, n, *ldo, out_type);
  return libxsmm_dispatch_meqn(idx, arg_shape);
}

inline int meqn_push_unary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_unary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  // OP metadata include equation id and an integer dictating where the op
  // metadata at runtime (if any) are located in the op arg array. -1 dictates
  // there are no op metadata needed
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_unary_op(op_metadata, type, dtype, flags);
}
inline int meqn_push_binary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_binary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_BINARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_binary_op(op_metadata, type, dtype, flags);
}
inline int meqn_push_ternary_op(
    const libxsmm_blasint idx,
    const libxsmm_meltw_ternary_type type,
    const libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_TERNARY_NONE,
    const libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32) {
  libxsmm_meqn_op_metadata op_metadata =
      libxsmm_create_meqn_op_metadata(idx, -1);
  return libxsmm_meqn_push_back_ternary_op(op_metadata, type, dtype, flags);
}

class BaseTPP {
 public:
  void* get_kernel() {
    // auto t0 = __rdtsc();
    auto& kernel_cache = get_kernel_cache();
    void* kernel = NULL;
    if (hash == "")
      hash = hash_str();
    // auto t1 = __rdtsc();
    auto search = kernel_cache.find(hash);
    if (search != kernel_cache.end())
      kernel = search->second;
    if (kernel == NULL) {
      kernel = build_kernel();
      if (kernel == NULL) {
        fprintf(stderr, "Unable to get JIT kernel for %s\n", hash.c_str());
        exit(1);
      }
      // printf("TPP: %s @ %p\n", hash.c_str(), kernel);
      kernel_cache[hash] = kernel;
      // printf("Hash size = %ld\n", (long)kernel_cache.size());
    }
    // auto t2 = __rdtsc();
    //  hsh_key += t1-t0;
    //  hsh_ret += t2-t1;
    // printf("%6lld  %6lld %6lld  get_kernel[%s]\n", t2-t0, (t1-t0), (t2-t1),
    // hash.c_str());
    return kernel;
  }
  // We should make hash_str() public
  std::string get_hash_str() {
    return hash_str();
  }

 protected:
#if 0
   std::unordered_map<std::string, void*>& get_kernel_cache() {
     static std::unordered_map<std::string, void*> kernel_cache;
     return kernel_cache;
   }
#else
  ska::flat_hash_map<std::string, void*>& get_kernel_cache() {
    static ska::flat_hash_map<std::string, void*> kernel_cache;
    return kernel_cache;
  }
#endif
  virtual std::string hash_str() = 0;
  virtual void* build_kernel() = 0;
  std::string hash = "";
  bool initialized = false;
};

class UnaryTPP : public BaseTPP {
 public:
  UnaryTPP() {}
  UnaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_unary_type type)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        dt_in(dt_in),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_unary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    kernel(&unary_param);
  }
  void operator()(void* in, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }
  void operator()(void* in, void* in2, void* in3, void* out, void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

  void operator()(
      void* in,
      void* in2,
      void* in3,
      void* op,
      void* op2,
      void* op3,
      void* out,
      void* out2) {
    if (!initialized)
      return;
    libxsmm_meltw_unary_param unary_param;
    unary_param.in.primary = in;
    unary_param.in.secondary = in2;
    unary_param.in.tertiary = in3;
    unary_param.op.primary = op;
    unary_param.op.secondary = op2;
    unary_param.op.tertiary = op3;
    unary_param.out.primary = out;
    unary_param.out.secondary = out2;
    kernel(&unary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "unary_r%d_c%d_i%d_o%d_di%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi,
        ldo,
        dt_in,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
        cols, rows, ldi, ldo, dt_in, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_unary(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi = 0;
  libxsmm_blasint ldo = 0;
  libxsmm_datatype dt_in = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_out = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype dt_compute = LIBXSMM_DATATYPE_F32;
  libxsmm_bitfield flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_meltwfunction_unary kernel = NULL;
};

class BinaryTPP : public BaseTPP {
 public:
  BinaryTPP() {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : BinaryTPP(
            rows,
            cols,
            ldi,
            ldi,
            ldo,
            dt_in,
            dt_in,
            dt_out,
            dt_compute,
            flags,
            type) {}
  BinaryTPP(
      libxsmm_blasint rows,
      libxsmm_blasint cols,
      libxsmm_blasint ldi0,
      libxsmm_blasint ldi1,
      libxsmm_blasint ldo,
      libxsmm_datatype dt_in0,
      libxsmm_datatype dt_in1,
      libxsmm_datatype dt_out,
      libxsmm_datatype dt_compute,
      libxsmm_bitfield flags,
      libxsmm_meltw_binary_type type)
      : rows(rows),
        cols(cols),
        ldi0(ldi0),
        ldi1(ldi1),
        ldo(ldo),
        dt_in0(dt_in0),
        dt_in1(dt_in1),
        dt_out(dt_out),
        dt_compute(dt_compute),
        flags(flags),
        type(type) {
    kernel = (libxsmm_meltwfunction_binary)get_kernel();
    if (kernel)
      initialized = true;
  }

  void operator()(void* in0, void* in1, void* out) {
    if (!initialized)
      return;
    libxsmm_meltw_binary_param binary_param;
    binary_param.in0.primary = in0;
    binary_param.in1.primary = in1;
    binary_param.out.primary = out;
    kernel(&binary_param);
  }

 protected:
  std::string hash_str() override {
    char hash[200];
    snprintf(
        hash,
        200,
        "binary_r%d_c%d_i0%d_i1%d_o%d_di0%d_di1%d_do%d_dc%d_f%d_t%d",
        rows,
        cols,
        ldi0,
        ldi1,
        ldo,
        dt_in0,
        dt_in1,
        dt_out,
        dt_compute,
        flags,
        type);
    return std::string(hash);
  }
  void* build_kernel() override {
    libxsmm_meltw_binary_shape shape = libxsmm_create_meltw_binary_shape(
        cols, rows, ldi0, ldi1, ldo, dt_in0, dt_in1, dt_out, dt_compute);
    return (void*)libxsmm_dispatch_meltw_binary(type, shape, flags);
  }

  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi0;
  libxsmm_blasint ldi1;
  libxsmm_blasint ldo;
  libxsmm_datatype dt_in0;
  libxsmm_datatype dt_in1;
  libxsmm_datatype dt_out;
  libxsmm_datatype dt_compute;
  libxsmm_bitfield flags;
  libxsmm_meltw_binary_type type;
  libxsmm_meltwfunction_binary kernel = NULL;
};

template <typename T>
class SetZeroTPP {
 public:
  SetZeroTPP() {}
  SetZeroTPP(int N) : SetZeroTPP(1, N) {}
  SetZeroTPP(int rows, int cols) : SetZeroTPP(rows, cols, cols) {}
  SetZeroTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_XOR) {}
  void operator()(T* buf) {
    kernel((void*)buf, (void*)buf);
  }
  void ref(T* buf) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        buf[i * ldo + j] = 0;
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout>
class ConvertTPP {
 public:
  ConvertTPP() {}
  ConvertTPP(int N) : ConvertTPP(1, N) {}
  ConvertTPP(int rows, int cols) : ConvertTPP(rows, cols, cols, cols) {}
  ConvertTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY),
        init_done(true) {}
  void operator()(Tin* in, Tout* out) {
    if (!(XsmmDtype<Tin>() == LIBXSMM_DATATYPE_F32 &&
          XsmmDtype<Tout>() == LIBXSMM_DATATYPE_F32) ||
        ((void*)in != (void*)out))
      kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[i * ldi + j];
      }
    }
  }
  bool initialized() {
    return init_done;
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
  bool init_done = false;
};

template <typename T>
class CpyTPP {
 public:
  CpyTPP() {}
  CpyTPP(int N) : CpyTPP(1, N) {}
  CpyTPP(int rows, int cols) : CpyTPP(rows, cols, cols, cols) {}
  CpyTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(T* in, T* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(T* in, T* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = in[i * ldi + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class CpyBiasTPP {
 public:
  CpyBiasTPP() {}
  CpyBiasTPP(int rows, int cols) : CpyBiasTPP(rows, cols, cols) {}
  CpyBiasTPP(int rows, int cols, int ldo)
      : rows(rows),
        cols(cols),
        ldo(ldo),
        kernel(
            rows,
            cols,
            cols,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>()
                                                  : LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL,
            LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i * ldo + j] = (Tout)in[j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldo;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class MulTPP {
 public:
  MulTPP() {}
  MulTPP(int N) : MulTPP(1, N) {}
  MulTPP(int rows, int cols) : MulTPP(rows, cols, cols, cols) {}
  MulTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(Tin* in0, Tin* in1, Tout* out) {
    kernel((void*)in0, (void*)in1, (void*)out);
  }
  void ref(Tin* in0, Tin* in1, Tout* out) {
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[r * ldo + c] = (float)in0[r * ldi + c] * (float)in1[r * ldi + c];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  BinaryTPP kernel;
};
class XformTPP {
 public:
  XformTPP() {}
  XformTPP(
      libxsmm_blasint rows_i,
      libxsmm_blasint cols_i,
      libxsmm_blasint ldi,
      libxsmm_blasint ldo,
      libxsmm_datatype dtype,
      libxsmm_meltw_unary_type type)
      : rows(rows_i),
        cols(cols_i),
        ldi(ldi),
        ldo(ldo),
        dtype(dtype),
        type(type),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            dtype,
            dtype,
            dtype,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            type) {}
  void operator()(void* in, void* out) {
    kernel(in, out);
  }
  typedef enum XFORM_TYPE {
    XFORM_NONE_TPP = 0,
    XFORM_XPOSE_TPP = 1,
    XFORM_N2V_TPP = 2,
    XFORM_XPOSE_N2V_TPP = 3,
    XFORM_XPOSE_V2V_TPP = 4
  } XFORM_TYPE;

 private:
  libxsmm_blasint rows = 0;
  libxsmm_blasint cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_datatype dtype;
  libxsmm_meltw_unary_type type;
  UnaryTPP kernel;
};

template <typename T>
class XformExtTPP {
 public:
  XformExtTPP() {}
  XformExtTPP(
      /* rows and cols as for input tensor */
      int rows,
      int cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            rows,
            cols,
            (xtype == XformTPP::XFORM_N2V_TPP ? rows : cols),
            (xtype == XformTPP::XFORM_N2V_TPP ? cols : rows),
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : XformExtTPP(
            in_rows,
            in_cols,
            out_rows,
            out_cols,
            in_cols,
            out_cols,
            xtype,
            ignore_vnni_for_fp32) {}
  XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      int ldi,
      int ldo,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
      : in_rows(in_rows),
        in_cols(in_cols),
        out_rows(out_rows),
        out_cols(out_cols),
        ldi(ldi),
        ldo(ldo),
        xtype(xtype),
        dtype(XsmmDtype<T>()),
        kernel(),
        cvt(),
        cpy(),
        zero() {
    libxsmm_meltw_unary_type unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
    if (ignore_vnni_for_fp32 == false) {
      TPP_ASSERT(
          (xtype == XformTPP::XFORM_XPOSE_TPP || dtype != LIBXSMM_DATATYPE_F32),
          "Only Transpose Xofrm supported for FP32 datatype, specified %d\n",
          (int)xtype);
    }
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_N2V_TPP) {
      in_rows_p = out_rows;
      in_cols_p = out_cols;
      TPP_ASSERT(in_rows_p % BS == 0, "N2VTPP: unaligned number of rows\n");
      if (BS == 1) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
      } else if (BS == 2) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
      } else if (BS == 4) {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
      } else {
        TPP_ASSERT(false, "N2VTPP: unsupported packing size (%d)\n", BS);
      }
    } else {
      in_rows_p = out_cols;
      in_cols_p = out_rows;
      if (dtype != LIBXSMM_DATATYPE_F32) {
        if (xtype == XformTPP::XFORM_XPOSE_TPP) {
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
        } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
          // unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT;
          unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
          TPP_ASSERT(
              in_cols_p % BS == 0, "XposeN2VTPP: uneven number of cols\n");
        } else {
          if (BS == 2) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T;
          } else if (BS == 4) {
            unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T;
          } else {
            TPP_ASSERT(false, "V2VTPP: unsupported packing size (%d)\n", BS);
          }

          TPP_ASSERT(in_rows % BS == 0, "XposeV2VTPP: uneven number of rows\n");
          TPP_ASSERT(
              in_cols_p % BS == 0, "XposeV2VTPP: uneven number of cols\n");
        }
      } else {
        unary_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;
      }
    }
    TPP_ASSERT(
        (in_rows_p >= in_rows && in_cols_p >= in_cols),
        "Invalid output rows or cols value\n");
    TPP_ASSERT(
        in_rows_p == in_rows || in_cols_p == in_cols,
        "Padding can only be done in rows or cols\n");

    if (xtype != XformTPP::XFORM_XPOSE_N2V_TPP) {
      int ld = (in_rows_p != in_rows || in_cols_p != in_cols) ? in_cols_p : ldi;
      kernel = XformTPP(in_rows_p, in_cols_p, ld, ldo, dtype, unary_type);
    } else {
      // LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT not implemented so use
      // workaround...
      kernel = XformTPP(
          in_rows_p,
          in_cols_p / BS,
          ldi / BS,
          ldo,
          ((dtype == LIBXSMM_DATATYPE_BF16 && BS == 4) ||
           (dtype == LIBXSMM_DATATYPE_BF8 && BS == 8))
              ? LIBXSMM_DATATYPE_F64
              : LIBXSMM_DATATYPE_F32,
          unary_type);
    }
    if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP && in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows / BS, in_cols * BS, ldi * BS, in_cols_p * BS);
      zero = SetZeroTPP<T>(
          in_rows / BS, (in_cols_p - in_cols) * BS, in_cols_p * BS);
      zero_offset = in_cols * BS;
    } else if (/*(xtype == XformTPP::XFORM_N2V_TPP ||
         xtype == XformTPP::XFORM_XPOSE_TPP) &&*/
               in_rows_p != in_rows) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols);
      zero = SetZeroTPP<T>(in_rows_p - in_rows, in_cols);
      zero_offset = in_rows * in_cols;
    } else if (
        /*xtype == XformTPP::XFORM_XPOSE_N2V_TPP &&*/ in_cols_p != in_cols) {
      cpy = CpyTPP<T>(in_rows, in_cols, ldi, in_cols_p);
      zero = SetZeroTPP<T>(in_rows, in_cols_p - in_cols, in_cols_p);
      zero_offset = in_cols;
    }
    if (std::is_same<T, bfloat16>::value)
      cvt = ConvertTPP<float, bfloat16>(in_rows, in_cols);
  }
  void operator()(T* in, T* out) {
    if (in != out) {
      if (in_rows_p != in_rows || in_cols_p != in_cols) {
        T tmp[in_rows_p * in_cols_p];
        cpy(in, tmp);
        zero(tmp + zero_offset);
        kernel((void*)tmp, (void*)out);
      } else {
        kernel((void*)in, (void*)out);
      }
    }
  }
  void ref(T* in, T* out) {
    const int BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < in_rows / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols && i * BS + k < out_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      TPP_ASSERT(false, "Should not come here\n");
    }
  }
  void operator()(float* in, bfloat16* out) {
    bfloat16 tmp2[in_rows * in_cols];
    cvt(in, tmp2);
    if (in_rows_p != in_rows || in_cols_p != in_cols) {
      T tmp[in_rows_p * in_cols_p];
      cpy(tmp2, tmp);
      zero(tmp + zero_offset);
      kernel((void*)tmp, (void*)out);
    } else {
      kernel((void*)tmp2, (void*)out);
    }
  }
  void ref(float* in, bfloat16* out) {
    auto BS = xsmm_get_vnni_block_size(dtype);
    if (xtype == XformTPP::XFORM_XPOSE_TPP) {
      for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
          out[i * ldo + j] = in[j * ldi + i];
        }
      }
    } else if (xtype == XformTPP::XFORM_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_rows) {
              out[i * ldo * BS + j * BS + k] = in[i * ldi * BS + k * ldi + j];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
      for (int i = 0; i < out_rows / BS; i++) {
        for (int j = 0; j < out_cols; j++) {
          for (int k = 0; k < BS; k++) {
            if (i * BS + k < in_cols) {
              out[i * ldo * BS + j * BS + k] = in[j * ldi + i * BS + k];
            } else {
              out[i * ldo * BS + j * BS + k] = 0;
            }
          }
        }
      }
    } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
      for (int j = 0; j < out_rows / BS; j++) {
        for (int i = 0; i < out_cols / BS; i++) {
          for (int k = 0; k < BS; k++) { // RBS
            for (int l = 0; l < BS; l++) { // CBS
              if (j * BS + l < in_cols) {
                out[j * ldo * BS + i * BS * BS + k * BS + l] =
                    in[i * ldi * BS + j * BS * BS + l * BS + k];
              } else {
                out[j * ldo * BS + i * BS * BS + k * BS + l] = 0;
              }
            }
          }
        }
      }
    } else {
      TPP_ASSERT(false, "Should not come here\n");
    }
  }
  void operator()(int count, long str_in, long str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(int count, long str_in, long str_out, T* in, T* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }
  void operator()(
      int count,
      long str_in,
      long str_out,
      float* in,
      bfloat16* out) {
    for (int i = 0; i < count; i++) {
      this->operator()(&in[i * str_in], &out[i * str_out]);
    }
  }
  void ref(int count, long str_in, long str_out, float* in, bfloat16* out) {
    for (int i = 0; i < count; i++) {
      this->ref(&in[i * str_in], &out[i * str_out]);
    }
  }

 private:
  libxsmm_blasint in_rows = 0;
  libxsmm_blasint in_cols = 0;
  libxsmm_blasint out_rows = 0;
  libxsmm_blasint out_cols = 0;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int in_rows_p = 0;
  int in_cols_p = 0;
  XformTPP::XFORM_TYPE xtype;
  libxsmm_datatype dtype;
  int zero_offset = 0;
  XformTPP kernel;
  ConvertTPP<float, bfloat16> cvt;
  CpyTPP<T> cpy;
  SetZeroTPP<T> zero;
};

template <typename Tin, typename Tout>
class BrgemmTPP {
 public:
  BrgemmTPP() {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      float beta = 1.0,
      int a_trans = 0,
      int unroll_hint = 0)
      : BrgemmTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            unroll_hint) {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      long lda,
      long ldb,
      long ldc,
      float beta,
      int a_trans,
      int unroll_hint,
      int b_vnni = 1)
      : M(M),
        N(N),
        K(K),
        str_a(str_a),
        str_b(str_b),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        beta(beta),
        a_trans(a_trans),
        unroll_hint(unroll_hint),
        b_vnni(b_vnni),
        k_gemm_with_tc(this, 0),
        k_cfg(this, 1),
        k_rls(this, 2),
        k_gemm_no_tc(this, 3) {}
  void config() {
    k_cfg(NULL);
  }
  void release() {
    k_rls(NULL);
  }
  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    libxsmm_gemm_param gemm_param;
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.op.tertiary = &count;
    gemm_param.c.primary = (void*)C;
    gemm_param.a.primary = (void*)B;
    gemm_param.b.primary = (void*)A;
    if (!no_tile_cfg) {
      k_gemm_with_tc(&gemm_param);
    } else {
      k_gemm_no_tc(&gemm_param);
    }
  }
  void ref(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {
    auto dtype = XsmmDtype<Tin>();
    for (uint64_t c = 0; c < count; c++) {
      auto A_ = &A[c * str_a];
      auto B_ = &B[c * str_b];
      if (std::is_same<Tin, float>::value || b_vnni == 0) {
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            if (beta == 0.0 && c == 0)
              C[i * N + j] = 0.0;
            for (int k = 0; k < K; k++) {
              if (a_trans == 1) {
                C[i * ldc + j] += A_[k * lda + i] * B_[k * ldb + j];
              } else {
                C[i * ldc + j] += A_[i * lda + k] * B_[k * ldb + j];
              }
            }
          }
        }
      } else {
        const int BS = xsmm_get_vnni_block_size(dtype);
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            float sum =
                ((beta == 0.0 && c == 0) ? 0.0f : (float)C[i * ldc + j]);
            for (int k = 0; k < K / BS; k++) {
              for (int b = 0; b < BS; b++) {
                if (a_trans == 1) {
                  sum += (float)A_[k * lda * BS + i * BS + b] *
                      (float)B_[k * ldb * BS + j * BS + b];
                } else {
                  sum += (float)A_[i * lda + k * BS + b] *
                      (float)B_[k * ldb * BS + j * BS + b];
                }
              }
            }
            C[i * ldc + j] = (Tout)sum;
          }
        }
      }
    }
  }

  long flops() {
    return 2L * M * N * K;
  }

  class BrgemmKernel : public BaseTPP {
   public:
    BrgemmKernel() {}
    BrgemmKernel(BrgemmTPP* p, int config) : p(p), config(config) {
      auto dt_in = XsmmDtype<Tin>();
      auto dt_out = XsmmDtype<Tout>();
      long type = -1;
      if (dt_in == LIBXSMM_DATATYPE_F32) {
        TPP_ASSERT(dt_out == LIBXSMM_DATATYPE_F32, "BRGEMM Assert\n");
        type = 0;
      } else if (dt_out == LIBXSMM_DATATYPE_F32) {
        type = 1;
      } else {
        type = 2;
      }
      brgemm_type = type;
      if (config == 1 || config == 2) {
        kernel.tilecfg = (libxsmm_tilecfgfunction)get_kernel();
      } else {
        kernel.gemm = (libxsmm_gemmfunction)get_kernel();
      }
      initialized = true;
    }
    void operator()(libxsmm_gemm_param* gemm_param) {
      if (!initialized)
        return;
      if (config == 1 || config == 2) {
        kernel.tilecfg(NULL);
      } else {
        kernel.gemm(gemm_param);
      }
    }

   protected:
    std::string hash_str() override {
      char hash[200];
      snprintf(
          hash,
          200,
          "brgemm_m%ld_n%ld_k%ld_a%ld_b%ld_t%ld_beta%d_at%d_uh%d_ld_a%ld_b%ld_c%ld_cfg%d_bv%d_dti%d_dto%d",
          p->M,
          p->N,
          p->K,
          p->str_a,
          p->str_b,
          brgemm_type,
          (int)p->beta,
          p->a_trans,
          p->unroll_hint,
          (long)p->lda,
          (long)p->ldb,
          (long)p->ldc,
          config,
          p->b_vnni,
          XsmmDtype<Tin>(),
          XsmmDtype<Tout>());
      return std::string(hash);
    }
    void* build_kernel() override {
      libxsmm_gemm_shape l_shape;
      libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');

      if (p->a_trans == 1)
        l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
      if (brgemm_type != 0) {
        if (p->b_vnni)
          l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
        if (p->a_trans == 1) {
          l_flags |= LIBXSMM_GEMM_FLAG_VNNI_B;
        }
      }
      if (p->beta == 0)
        l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

      l_shape.m = p->N;
      l_shape.n = p->M;
      l_shape.k = p->K;
      l_shape.lda = p->ldb;
      l_shape.ldb = p->lda;
      l_shape.ldc = p->ldc;
      l_shape.a_in_type = XsmmDtype<Tin>();
      l_shape.b_in_type = XsmmDtype<Tin>();
      l_shape.out_type = XsmmDtype<Tout>();
      l_shape.comp_type = LIBXSMM_DATATYPE_F32;

      if (config == 1 || config == 2) {
        // config 1: setup tile config only (no reset)
        // config 2: release tile config only (no setup)
        if (config == 1) {
          l_flags |= LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
        } else {
          l_flags |= LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
        }
        return (void*)libxsmm_dispatch_tilecfg_gemm(l_shape, l_flags);
      }

      // config 0: normal brgemm (with tile config + release)
      // config 3: brgemm with no tile config or release
      libxsmm_gemm_batch_reduce_config l_brconfig;
      libxsmm_bitfield l_prefetch_flags = 0;
      libxsmm_xmmfunction l_test_jit = {NULL};

      if (config == 3) {
        l_flags |=
            (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
             LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
      }

      l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
      l_brconfig.br_stride_a_hint = p->str_b * sizeof(Tin);
      l_brconfig.br_stride_b_hint = p->str_a * sizeof(Tin);
      l_brconfig.br_unroll_hint = p->unroll_hint;

      l_test_jit.gemm = libxsmm_dispatch_brgemm(
          l_shape, l_flags, l_prefetch_flags, l_brconfig);

      return (void*)l_test_jit.gemm;
    }

   private:
    BrgemmTPP* p;
    int config;
    libxsmm_xmmfunction kernel;
    long brgemm_type = -1;
  };

 private:
  long M, N, K, str_a, str_b;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  float beta;
  int a_trans;
  long brgemm_type = -1;
  int unroll_hint;
  int b_vnni;
  BrgemmKernel k_gemm_with_tc;
  BrgemmKernel k_cfg;
  BrgemmKernel k_rls;
  BrgemmKernel k_gemm_no_tc;
};

template <typename Tin, typename Tout = Tin>
class GeluFwdTPP {
 public:
  GeluFwdTPP() {}
  GeluFwdTPP(int N) : GeluFwdTPP(1, N) {}
  GeluFwdTPP(int M, int N) : GeluFwdTPP(M, N, N, N) {}
  GeluFwdTPP(int M, int N, int ldi, int ldo)
      : M(M),
        N(N),
        ldi(ldi),
        ldo(ldo),
        kernel(
            M,
            N,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_GELU) {}
  void operator()(Tin* in, Tout* out) {
    kernel((void*)in, (void*)out);
  }
  void ref(Tin* in, Tout* out) {
#ifdef __AVX512F__
    for (int j = 0; j < M; j++) {
      int i;
      for (i = 0; i < ALIGNDOWN(N, 16); i += 16) {
        auto vin = _mm512_loadu_ps_auto(&in[j * ldi + i]);
        // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
        auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
        _mm512_storeu_ps_auto(&out[j * ldo + i], vout);
      }
      if (i < N) {
        int rem = N - i;
        __mmask16 mask = (1 << rem) - 1;
        auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[j * ldi + i]);
        // auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
        auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
        _mm512_mask_storeu_ps_auto(&out[j * ldo + i], mask, vout);
      }
    }
#else
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < N; i++) {
        float x = in[j * ldi + i];
        out[j * ldo + i] = (erff(x / sqrtf(2.0)) + 1.0) * 0.5 * x;
      }
    }
#endif
  }

 private:
  int M = 0;
  int N = 0;
  int ldi = 0;
  int ldo = 0;
  UnaryTPP kernel;
};

template <typename Tin, typename Tout = Tin>
class ReLUFwdTPP {
 public:
  ReLUFwdTPP() {}
  ReLUFwdTPP(int N, bool bm) : ReLUFwdTPP(1, N, bm) {}
  ReLUFwdTPP(int rows, int cols, bool bm)
      : ReLUFwdTPP(rows, cols, cols, cols, bm) {}
  ReLUFwdTPP(int rows, int cols, int ldi, int ldo, bool bm = false)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        kernel(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<Tin>(),
            XsmmDtype<Tout>(),
            LIBXSMM_DATATYPE_F32,
            bm ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT
               : LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_RELU) {}
  void operator()(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }
  void ref(Tin* in, Tout* out, short* mask = NULL) {
    kernel((void*)in, (void*)out, (void*)mask);
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP kernel;
};

template <typename T>
class SiLUFwdTPP {
 public:
  SiLUFwdTPP() {}
  SiLUFwdTPP(int N) : SiLUFwdTPP(1, N) {}
  SiLUFwdTPP(int rows, int cols) : SiLUFwdTPP(rows, cols, cols, cols) {}
  SiLUFwdTPP(int rows, int cols, int ldi, int ldo)
      : rows(rows),
        cols(cols),
        ldi(ldi),
        ldo(ldo),
        sigmoid(
            rows,
            cols,
            ldi,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE,
            LIBXSMM_MELTW_TYPE_UNARY_SIGMOID),
        mul(rows,
            cols,
            ldi,
            ldo,
            ldo,
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            XsmmDtype<T>(),
            LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_BINARY_NONE,
            LIBXSMM_MELTW_TYPE_BINARY_MUL) {}
  void operator()(T* in, T* out, T* sigout = nullptr) {
    T tmp[rows * ldo];
    if (sigout == nullptr)
      sigout = tmp;
    sigmoid((void*)in, (void*)sigout);
    mul((void*)in, (void*)sigout, (void*)out);
  }
  void ref(T* in, T* out, T* sigout = nullptr) {
    T tmp[rows * ldo];
    if (sigout == nullptr)
      sigout = tmp;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        sigout[i * ldo + j] = 1. / (1. + exp(-in[i * ldi + j]));
        out[i * ldo + j] = in[i * ldi + j] * sigout[i * ldo + j];
      }
    }
  }

 private:
  int rows = 0;
  int cols = 0;
  int ldi;
  int ldo;
  UnaryTPP sigmoid;
  BinaryTPP mul;
};
}

; // namespace tpp

#endif // _XSMM_FUNCTORS_H_
