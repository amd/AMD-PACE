/******************************************************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 ******************************************************************************************************************/

/******************************************************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.
 *
 * For information on the license, see the LICENSE file. Further information:
 * https://github.com/libxsmm/tpp-pytorch-extension/
 * Source Code:
 * https://github.com/libxsmm/tpp-pytorch-extension/blob/mlperf_infer_31/src/csrc/llm/fused_llm_infer.cpp
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************************************************/

/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************************************************/

#include <omp.h>
#include <ops/kernels/libxsmm_linear_kernel.h>
#include <ops/libxsmm_dependency/threaded_loops.h>
#include <ops/libxsmm_dependency/utils.h>
#include <ops/libxsmm_dependency/xsmm_functors.h>
#include <mutex>
namespace pace {

namespace kernels {
static int PACE_LARGE_CACHE_OPT = false;
static int PACE_NCB_BLOCK_SIZE = env2int("PACE_NCB_BLOCK_SIZE", 64);
static int PACE_DOWN_NCB = env2int("PACE_DOWN_NCB", 32);
static int PACE_MLP_BSB = env2int("PACE_MLP_BSB", 0);
static const char* PACE_GEMM_LOOP_SCHEME =
    getenv("PACE_GEMM_LOOP_SCHEME") ? getenv("PACE_GEMM_LOOP_SCHEME") : "aCB";

using DataType = at::BFloat16;

namespace impl {

inline size_t hash_combine(size_t seed, size_t v) {
  return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

template <typename Key, typename Value, typename Hash>
class TPPCache {
  std::unordered_map<Key, Value, Hash> map_;
  std::mutex mutex_;

 public:
  template <typename Factory>
  Value& get_or_create(const Key& key, Factory&& factory) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = map_.find(key);
      if (it != map_.end())
        return it->second;
    }
    Value val = factory(key);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = map_.find(key);
      if (it != map_.end())
        return it->second;
      auto res = map_.emplace(key, std::move(val));
      return res.first->second;
    }
  }
};

// ---- Linear kernel cache ----

struct LinearKey {
  long BSb, Hk, Hc, C, K, Ncb, rem;
  bool operator==(const LinearKey& o) const {
    return BSb == o.BSb && Hk == o.Hk && Hc == o.Hc && C == o.C && K == o.K &&
        Ncb == o.Ncb && rem == o.rem;
  }
};

struct LinearKeyHash {
  size_t operator()(const LinearKey& k) const {
    size_t h = 0;
    for (auto v : {k.BSb, k.Hk, k.Hc, k.C, k.K, k.Ncb, k.rem})
      h = hash_combine(h, std::hash<long>()(v));
    return h;
  }
};

struct LinearTPPs {
  BrgemmTPP<DataType, DataType> brgemm, brgemm_rem;
  CpyBiasTPP<DataType> copy_bias, copy_bias_rem;
  SetZeroTPP<DataType> zero, zero_rem;
  ReLUActivation relu, relu_rem;
  GeluActivation gelu, gelu_rem;
  SiLUActivation silu, silu_rem;
  MulActivation mul, mul_rem;
};

static TPPCache<LinearKey, LinearTPPs, LinearKeyHash> linear_cache;

static LinearTPPs& get_linear_tpps(const LinearKey& key) {
  return linear_cache.get_or_create(key, [](const LinearKey& k) {
    auto BSb = k.BSb, Hk = k.Hk, Hc = k.Hc, C = k.C;
    auto K = k.K, Ncb = k.Ncb, rem = k.rem;
    LinearTPPs t;
    t.brgemm = BrgemmTPP<DataType, DataType>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb);
    t.brgemm_rem = (rem > 0)
        ? BrgemmTPP<DataType, DataType>(
              rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb)
        : BrgemmTPP<DataType, DataType>();
    t.copy_bias = CpyBiasTPP<DataType>(BSb, Hk, K);
    t.copy_bias_rem =
        (rem > 0) ? CpyBiasTPP<DataType>(rem, Hk, K) : CpyBiasTPP<DataType>();
    t.zero = SetZeroTPP<DataType>(BSb, Hk, K);
    t.zero_rem =
        (rem > 0) ? SetZeroTPP<DataType>(rem, Hk, K) : SetZeroTPP<DataType>();
    t.relu = ReLUActivation(BSb, Hk, K, K);
    t.relu_rem =
        (rem > 0) ? ReLUActivation(rem, Hk, K, K) : ReLUActivation(0, 0, 0, 0);
    t.gelu = GeluActivation(BSb, Hk, K, K);
    t.gelu_rem =
        (rem > 0) ? GeluActivation(rem, Hk, K, K) : GeluActivation(0, 0, 0, 0);
    t.silu = SiLUActivation(BSb, Hk, K, K);
    t.silu_rem =
        (rem > 0) ? SiLUActivation(rem, Hk, K, K) : SiLUActivation(0, 0, 0, 0);
    t.mul = MulActivation(BSb, Hk, K, K);
    t.mul_rem =
        (rem > 0) ? MulActivation(rem, Hk, K, K) : MulActivation(0, 0, 0, 0);
    return t;
  });
}

template <typename ActivationTPP>
struct LinearActSelector;

template <>
struct LinearActSelector<ReLUActivation> {
  static ReLUActivation& get(LinearTPPs& t, bool r) {
    return r ? t.relu_rem : t.relu;
  }
};
template <>
struct LinearActSelector<GeluActivation> {
  static GeluActivation& get(LinearTPPs& t, bool r) {
    return r ? t.gelu_rem : t.gelu;
  }
};
template <>
struct LinearActSelector<SiLUActivation> {
  static SiLUActivation& get(LinearTPPs& t, bool r) {
    return r ? t.silu_rem : t.silu;
  }
};
template <>
struct LinearActSelector<MulActivation> {
  static MulActivation& get(LinearTPPs& t, bool r) {
    return r ? t.mul_rem : t.mul;
  }
};
template <>
struct LinearActSelector<NoOpActivation> {
  static NoOpActivation& get(LinearTPPs&, bool) {
    static NoOpActivation noop(0, 0, 0, 0);
    return noop;
  }
};

// ---- Fused MLP kernel cache ----

struct FusedMLPKey {
  long BSb, Hk, Hc, C, N, K_out, Nc, Nc_d, Hc_d, Hk_d, Ncb_d, rem;
  bool operator==(const FusedMLPKey& o) const {
    return BSb == o.BSb && Hk == o.Hk && Hc == o.Hc && C == o.C && N == o.N &&
        K_out == o.K_out && Nc == o.Nc && Nc_d == o.Nc_d && Hc_d == o.Hc_d &&
        Hk_d == o.Hk_d && Ncb_d == o.Ncb_d && rem == o.rem;
  }
};

struct FusedMLPKeyHash {
  size_t operator()(const FusedMLPKey& k) const {
    size_t h = 0;
    for (auto v :
         {k.BSb,
          k.Hk,
          k.Hc,
          k.C,
          k.N,
          k.K_out,
          k.Nc,
          k.Nc_d,
          k.Hc_d,
          k.Hk_d,
          k.Ncb_d,
          k.rem})
      h = hash_combine(h, std::hash<long>()(v));
    return h;
  }
};

struct FusedMLPTPPs {
  BrgemmTPP<DataType, DataType> brgemm_gu, brgemm_gu_rem;
  BrgemmTPP<DataType, DataType> brgemm_gu_bias, brgemm_gu_bias_rem;
  CpyBiasTPP<DataType> cpbias_gu, cpbias_gu_rem;
  SiLUFwdTPP<DataType> silu, silu_rem;
  GeluFwdTPP<DataType> gelu, gelu_rem;
  ReLUFwdTPP<DataType> relu, relu_rem;
  MulTPP<DataType, DataType> mul, mul_rem;
  CpyTPP<DataType> cpy, cpy_rem;
  BrgemmTPP<DataType, DataType> brgemm_down, brgemm_down_rem;
  SetZeroTPP<DataType> zero_d, zero_d_rem;
  CpyBiasTPP<DataType> cpbias_d, cpbias_d_rem;
};

static TPPCache<FusedMLPKey, FusedMLPTPPs, FusedMLPKeyHash> fused_mlp_cache;

static FusedMLPTPPs& get_fused_mlp_tpps(const FusedMLPKey& key) {
  return fused_mlp_cache.get_or_create(key, [](const FusedMLPKey& k) {
    auto BSb = k.BSb, Hk = k.Hk, Hc = k.Hc, C = k.C;
    auto N = k.N, K_out = k.K_out, Nc = k.Nc;
    auto Hc_d = k.Hc_d, Hk_d = k.Hk_d;
    auto Ncb_d = k.Ncb_d, rem = k.rem;
    FusedMLPTPPs t;
    t.brgemm_gu = BrgemmTPP<DataType, DataType>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, Nc);
    t.brgemm_gu_rem = (rem > 0)
        ? BrgemmTPP<DataType, DataType>(
              rem, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 0.0, 0, Nc)
        : BrgemmTPP<DataType, DataType>();
    t.brgemm_gu_bias = BrgemmTPP<DataType, DataType>(
        BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 1.0, 0, Nc);
    t.brgemm_gu_bias_rem = (rem > 0)
        ? BrgemmTPP<DataType, DataType>(
              rem, Hk, Hc, Hc, Hk * Hc, C, Hk, Hk, 1.0, 0, Nc)
        : BrgemmTPP<DataType, DataType>();
    t.cpbias_gu = CpyBiasTPP<DataType>(BSb, Hk, Hk);
    t.cpbias_gu_rem =
        (rem > 0) ? CpyBiasTPP<DataType>(rem, Hk, Hk) : CpyBiasTPP<DataType>();
    t.silu = SiLUFwdTPP<DataType>(BSb, Hk, Hk, Hk);
    t.silu_rem = (rem > 0) ? SiLUFwdTPP<DataType>(rem, Hk, Hk, Hk)
                           : SiLUFwdTPP<DataType>();
    t.gelu = GeluFwdTPP<DataType>(BSb, Hk, Hk, Hk);
    t.gelu_rem = (rem > 0) ? GeluFwdTPP<DataType>(rem, Hk, Hk, Hk)
                           : GeluFwdTPP<DataType>();
    t.relu = ReLUFwdTPP<DataType>(BSb, Hk, Hk, Hk);
    t.relu_rem = (rem > 0) ? ReLUFwdTPP<DataType>(rem, Hk, Hk, Hk)
                           : ReLUFwdTPP<DataType>();
    t.mul = MulTPP<DataType, DataType>(BSb, Hk, Hk, Hk);
    t.mul_rem = (rem > 0) ? MulTPP<DataType, DataType>(rem, Hk, Hk, Hk)
                          : MulTPP<DataType, DataType>();
    t.cpy = CpyTPP<DataType>(BSb, Hk, Hk, N);
    t.cpy_rem =
        (rem > 0) ? CpyTPP<DataType>(rem, Hk, Hk, N) : CpyTPP<DataType>();
    t.brgemm_down = BrgemmTPP<DataType, DataType>(
        BSb, Hk_d, Hc_d, Hc_d, Hk_d * Hc_d, N, Hk_d, K_out, 1.0, 0, Ncb_d);
    t.brgemm_down_rem = (rem > 0)
        ? BrgemmTPP<DataType, DataType>(
              rem, Hk_d, Hc_d, Hc_d, Hk_d * Hc_d, N, Hk_d, K_out, 1.0, 0, Ncb_d)
        : BrgemmTPP<DataType, DataType>();
    t.zero_d = SetZeroTPP<DataType>(BSb, Hk_d, K_out);
    t.zero_d_rem = (rem > 0) ? SetZeroTPP<DataType>(rem, Hk_d, K_out)
                             : SetZeroTPP<DataType>();
    t.cpbias_d = CpyBiasTPP<DataType>(BSb, Hk_d, K_out);
    t.cpbias_d_rem = (rem > 0) ? CpyBiasTPP<DataType>(rem, Hk_d, K_out)
                               : CpyBiasTPP<DataType>();
    return t;
  });
}

static inline bool has_bias(const c10::optional<at::Tensor>& b) {
  return b.has_value() && b.value().numel() > 0 && b.value().dim() > 0;
}

template <typename ActTPP>
static inline void apply_act(ActTPP& act, DataType* g, DataType* o) {
  if constexpr (std::is_same_v<ActTPP, ReLUFwdTPP<DataType>>)
    act(g, o, nullptr);
  else
    act(g, o);
}

template <typename GateActivationTPP>
struct ActSelector;

template <>
struct ActSelector<SiLUFwdTPP<float>> {
  using type = SiLUFwdTPP<DataType>;
  static type& get(FusedMLPTPPs& t, bool r) {
    return r ? t.silu_rem : t.silu;
  }
};
template <>
struct ActSelector<GeluFwdTPP<float>> {
  using type = GeluFwdTPP<DataType>;
  static type& get(FusedMLPTPPs& t, bool r) {
    return r ? t.gelu_rem : t.gelu;
  }
};
template <>
struct ActSelector<ReLUFwdTPP<float>> {
  using type = ReLUFwdTPP<DataType>;
  static type& get(FusedMLPTPPs& t, bool r) {
    return r ? t.relu_rem : t.relu;
  }
};

} // namespace impl

template <typename ActivationTPP>
static inline void apply_activation(
    ActivationTPP& activation_tpp,
    at::BFloat16* in1,
    at::BFloat16* out) {
  if constexpr (std::is_same_v<ActivationTPP, MulActivation>) {
    activation_tpp(in1, out, out);
  } else {
    activation_tpp(out, out);
  }
}

template <typename ActivationTPP>
void libxsmmlinear_kernel(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];
  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;
  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);
  bool with_bias = (t_bias.dim() > 0);
  auto in = GetVLAPtr<DataType>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<DataType>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<DataType>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<DataType>(t_bias, {Hk});
  auto out = GetVLAPtr<DataType>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (PACE_LARGE_CACHE_OPT)
    Ncb = PACE_NCB_BLOCK_SIZE;

  impl::LinearKey key{BSb, Hk, Hc, C, K, Ncb, rem};
  auto& tpps = impl::get_linear_tpps(key);

  {
    auto loop_scheme = PACE_GEMM_LOOP_SCHEME;
    auto igemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop([&](int* ind) {
      int nc = ind[0], s1 = ind[1], nk = ind[2];
      auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
      bool is_rem = (s1 + BSb > BS);
      auto& brg = is_rem ? tpps.brgemm_rem : tpps.brgemm;

      if (nc == 0) {
        if (with_bias) {
          auto& cb = is_rem ? tpps.copy_bias_rem : tpps.copy_bias;
          cb(bias[nk], out[s1][nk]);
        } else {
          auto& z = is_rem ? tpps.zero_rem : tpps.zero;
          z(out[s1][nk]);
        }
      }
      brg(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, !is_rem);
      if (!(nc + Ncb < Nc)) {
        auto& act = impl::LinearActSelector<ActivationTPP>::get(tpps, is_rem);
        apply_activation<ActivationTPP>(act, in1[s1][nk], out[s1][nk]);
      }
    });
  }
}

#define INSTANTIATE_LIBXSMM_KERNEL(ActivationType)    \
  template void libxsmmlinear_kernel<ActivationType>( \
      at::Tensor & t_in,                              \
      at::Tensor & t_in1,                             \
      at::Tensor & t_wt,                              \
      at::Tensor & t_bias,                            \
      at::Tensor & t_out);

INSTANTIATE_LIBXSMM_KERNEL(ReLUActivation)
INSTANTIATE_LIBXSMM_KERNEL(GeluActivation)
INSTANTIATE_LIBXSMM_KERNEL(SiLUActivation)
INSTANTIATE_LIBXSMM_KERNEL(MulActivation)
INSTANTIATE_LIBXSMM_KERNEL(NoOpActivation)

#undef INSTANTIATE_LIBXSMM_KERNEL

namespace impl {

template <typename GateActivationTPP>
at::Tensor fused_mlp_kernel(
    const at::Tensor& t_in,
    const c10::optional<at::Tensor>& t_wt_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_wt_down,
    const c10::optional<at::Tensor>& t_gate_bias,
    const c10::optional<at::Tensor>& t_up_bias,
    const c10::optional<at::Tensor>& t_down_bias) {
  bool gated = t_wt_gate.has_value() && t_wt_gate.value().numel() > 0;

  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto C = in_sizes[2];

  const auto& wt_gu_ref = gated ? t_wt_gate.value() : t_wt_up;
  auto wt_gu_sizes = wt_gu_ref.sizes();
  auto Nc = wt_gu_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_gu_sizes[0];
  auto Hk = wt_gu_sizes[3];
  auto N = Nk * Hk;

  auto wt_d_sizes = t_wt_down.sizes();
  auto Nk_d = wt_d_sizes[0];
  auto Nc_d = wt_d_sizes[1];
  auto Hk_d = wt_d_sizes[3];
  auto K_out = Nk_d * Hk_d;
  auto Hc_d = N / Nc_d;

  long BSb;
  if (PACE_MLP_BSB > 0) {
    BSb = (long)PACE_MLP_BSB;
  } else {
    BSb = (BS >= 512) ? 128L : 64L;
  }
  auto rem = BS % BSb;
  auto BS_blocks = (BS + BSb - 1) / BSb;
  auto Ncb_d = std::min(Nc_d, (long)PACE_DOWN_NCB);

  FusedMLPKey key{BSb, Hk, Hc, C, N, K_out, Nc, Nc_d, Hc_d, Hk_d, Ncb_d, rem};
  auto& tpps = get_fused_mlp_tpps(key);

  auto t_inter = at::empty({in_sizes[0], in_sizes[1], N}, t_in.options());
  auto t_out = at::empty({in_sizes[0], in_sizes[1], K_out}, t_in.options());

  auto in = GetVLAPtr<DataType>(t_in, {Nc, Hc});
  auto empty_wt = gated ? t_wt_gate.value() : t_wt_up;
  auto wt_gate = GetVLAPtr<DataType>(empty_wt, {Nc, Hc * Hk});
  auto wt_up = GetVLAPtr<DataType>(t_wt_up, {Nc, Hc * Hk});
  auto inter = GetVLAPtr<DataType>(t_inter, {Nk, Hk});
  auto inter_d = GetVLAPtr<DataType>(t_inter, {Nc_d, Hc_d});
  auto wt_down = GetVLAPtr<DataType>(t_wt_down, {Nc_d, Hc_d * Hk_d});
  auto out = GetVLAPtr<DataType>(t_out, {Nk_d, Hk_d});

  auto empty_bf16 = at::empty({}, t_in.options());

  bool with_gate_bias = gated && has_bias(t_gate_bias);
  bool with_up_bias = has_bias(t_up_bias);
  bool with_down_bias = has_bias(t_down_bias);

  auto gb_t = with_gate_bias ? t_gate_bias.value() : empty_bf16;
  auto ub_t = with_up_bias ? t_up_bias.value() : empty_bf16;
  auto db_t = with_down_bias ? t_down_bias.value() : empty_bf16;
  auto gbias = GetVLAPtr<DataType>(gb_t, {Hk});
  auto ubias = GetVLAPtr<DataType>(ub_t, {Hk});
  auto down_bias = GetVLAPtr<DataType>(db_t, {Hk_d});

  int total_tiles = BS_blocks * Nk;

  long tile_elems = BSb * Hk;
  int max_threads = omp_get_max_threads();
  auto t_tile_buf =
      at::empty({max_threads, gated ? 2L : 1L, tile_elems}, t_in.options());
  auto* tile_buf_ptr = t_tile_buf.data_ptr<DataType>();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthr = omp_get_num_threads();
    int tpt = (total_tiles + nthr - 1) / nthr;
    int tile_start = tid * tpt;
    int tile_end = std::min(tile_start + tpt, total_tiles);

    DataType* tile_a = tile_buf_ptr + tid * (gated ? 2L : 1L) * tile_elems;
    DataType* tile_b = gated ? tile_a + tile_elems : nullptr;

    // Phase 1: gate + up projections, activation, element-wise multiply
    for (int tile = tile_start; tile < tile_end; tile++) {
      int nk = tile / BS_blocks;
      int s1 = (tile % BS_blocks) * BSb;
      bool is_rem = (s1 + BSb > BS);

      if (tile + 1 < tile_end) {
        int next_nk = (tile + 1) / BS_blocks;
        if (next_nk != nk) {
          if (gated)
            _mm_prefetch((const char*)wt_gate[next_nk][0], _MM_HINT_T2);
          _mm_prefetch((const char*)wt_up[next_nk][0], _MM_HINT_T2);
        }
      }

      auto& act = ActSelector<GateActivationTPP>::get(tpps, is_rem);
      auto& cp = is_rem ? tpps.cpy_rem : tpps.cpy;

      auto project = [&](DataType* tile,
                         DataType* wt_nk,
                         DataType* bias_nk,
                         bool with_bias) {
        if (with_bias) {
          auto& cb = is_rem ? tpps.cpbias_gu_rem : tpps.cpbias_gu;
          auto& brg_b = is_rem ? tpps.brgemm_gu_bias_rem : tpps.brgemm_gu_bias;
          cb(bias_nk, tile);
          brg_b(in[s1][0], wt_nk, tile, Nc, !is_rem);
        } else {
          auto& brg = is_rem ? tpps.brgemm_gu_rem : tpps.brgemm_gu;
          brg(in[s1][0], wt_nk, tile, Nc, !is_rem);
        }
      };

      if (gated) {
        project(
            tile_a,
            wt_gate[nk][0],
            with_gate_bias ? gbias[nk] : nullptr,
            with_gate_bias);
        project(
            tile_b,
            wt_up[nk][0],
            with_up_bias ? ubias[nk] : nullptr,
            with_up_bias);
        apply_act(act, tile_a, tile_a);
        auto& m = is_rem ? tpps.mul_rem : tpps.mul;
        m(tile_b, tile_a, tile_a);
      } else {
        project(
            tile_a,
            wt_up[nk][0],
            with_up_bias ? ubias[nk] : nullptr,
            with_up_bias);
        apply_act(act, tile_a, tile_a);
      }
      cp(tile_a, inter[s1][nk]);
    }

#pragma omp barrier

    // Phase 2: down projection with L2-aware sub-batching
    for (long nc_d = 0; nc_d < Nc_d; nc_d += Ncb_d) {
      auto count_d = std::min(Ncb_d, Nc_d - nc_d);
#pragma omp for collapse(2) schedule(static) nowait
      for (long nk_d = 0; nk_d < Nk_d; nk_d++) {
        for (long s1 = 0; s1 < BS; s1 += BSb) {
          bool is_rem = (s1 + BSb > BS);
          // TODO: upgrade prefetch hints to T0/T1 if they help.
          if (nc_d + Ncb_d < Nc_d) {
            _mm_prefetch((const char*)inter_d[s1][nc_d + Ncb_d], _MM_HINT_T1);
            _mm_prefetch((const char*)wt_down[nk_d][nc_d + Ncb_d], _MM_HINT_T2);
          }
          auto& brg_d = is_rem ? tpps.brgemm_down_rem : tpps.brgemm_down;
          if (nc_d == 0) {
            if (with_down_bias) {
              auto& cb_d = is_rem ? tpps.cpbias_d_rem : tpps.cpbias_d;
              cb_d(down_bias[nk_d], out[s1][nk_d]);
            } else {
              auto& z_d = is_rem ? tpps.zero_d_rem : tpps.zero_d;
              z_d(out[s1][nk_d]);
            }
          }
          brg_d(
              inter_d[s1][nc_d],
              wt_down[nk_d][nc_d],
              out[s1][nk_d],
              count_d,
              !is_rem);
        }
      }
    }
  }

  return t_out;
}

#define INSTANTIATE_FUSED_MLP(ActType)           \
  template at::Tensor fused_mlp_kernel<ActType>( \
      const at::Tensor&,                         \
      const c10::optional<at::Tensor>&,          \
      const at::Tensor&,                         \
      const at::Tensor&,                         \
      const c10::optional<at::Tensor>&,          \
      const c10::optional<at::Tensor>&,          \
      const c10::optional<at::Tensor>&);

INSTANTIATE_FUSED_MLP(SiLUFwdTPP<float>)
INSTANTIATE_FUSED_MLP(GeluFwdTPP<float>)
INSTANTIATE_FUSED_MLP(ReLUFwdTPP<float>)

#undef INSTANTIATE_FUSED_MLP

} // namespace impl

at::Tensor fused_mlp_dispatch(
    const at::Tensor& t_in,
    const c10::optional<at::Tensor>& t_wt_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_wt_down,
    const c10::optional<at::Tensor>& t_gate_bias,
    const c10::optional<at::Tensor>& t_up_bias,
    const c10::optional<at::Tensor>& t_down_bias,
    const std::string& activation) {
#define DISPATCH_ACT(Act)      \
  impl::fused_mlp_kernel<Act>( \
      t_in,                    \
      t_wt_gate,               \
      t_wt_up,                 \
      t_wt_down,               \
      t_gate_bias,             \
      t_up_bias,               \
      t_down_bias)

  if (activation == "silu")
    return DISPATCH_ACT(SiLUFwdTPP<float>);
  if (activation == "gelu")
    return DISPATCH_ACT(GeluFwdTPP<float>);
  if (activation == "relu")
    return DISPATCH_ACT(ReLUFwdTPP<float>);
  TORCH_CHECK(false, "fused_mlp: unsupported activation: ", activation);

#undef DISPATCH_ACT
}

} // namespace kernels

} // namespace pace
