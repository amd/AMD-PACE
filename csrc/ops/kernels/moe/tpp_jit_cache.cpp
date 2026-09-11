// ******************************************************************************
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.
// Portions of this file consist of AI-generated content
// ******************************************************************************

#include "moe.h"

namespace pace {
namespace kernels {

int PACE_MLP_BSB = env2int("PACE_MLP_BSB", 0);
int PACE_DOWN_NCB = env2int("PACE_DOWN_NCB", 32);

// TPP cache
template <int N>
class TPPCacheLocal {
  std::unordered_map<MoEMLPKey, MoEMLPTPPs, MoEMLPKeyHash> map_;
  std::mutex mutex_;

 public:
  // Reserve upfront to prevent rehash and reference invalidation.
  // Capacity: 512 distinct keys (num_experts * distinct_rem_values) is a safe
  // upper bound for any realistic MoE model shape.
  TPPCacheLocal() {
    map_.reserve(512);
  }

  template <typename Factory>
  MoEMLPTPPs& get_or_create(const MoEMLPKey& key, Factory&& factory) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = map_.find(key);
      if (it != map_.end())
        return it->second;
    }
    MoEMLPTPPs val = factory(key);
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

static TPPCacheLocal<0> moe_mlp_cache;

MoEMLPTPPs& get_moe_mlp_tpps(const MoEMLPKey& key) {
  return moe_mlp_cache.get_or_create(key, [](const MoEMLPKey& k) {
    auto BSb = k.BSb, Hk = k.Hk, Hc = k.Hc, C = k.C;
    auto N = k.N, K_out = k.K_out, Nc = k.Nc;
    auto Hc_d = k.Hc_d, Hk_d = k.Hk_d;
    auto Ncb_d = k.Ncb_d, rem = k.rem;
    MoEMLPTPPs t;
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

bool has_bias(const c10::optional<at::Tensor>& b) {
  return b.has_value() && b.value().numel() > 0 && b.value().dim() > 0;
}

} // namespace kernels
} // namespace pace
