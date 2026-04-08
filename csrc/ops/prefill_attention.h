/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_PREFILL_ATTENTION_H
#define PACE_PREFILL_ATTENTION_H

#include <ATen/ATen.h>
#include <vector>

namespace pace {

/**
 * BRGeMM tiled prefill attention for contiguous BMC cache tensors.
 *
 * @param query      [B, N_q, S_padded, H] BFloat16
 * @param key        [B, N_kv, KV_padded, H] BFloat16
 * @param value      [B, N_kv, KV_padded, H] BFloat16
 * @param q_offsets  [B] start index of real tokens per sequence (empty = no
 * padding)
 * @param q_lens     [B] real token count per sequence (empty = no padding)
 * @return output    [B, N_q, S_padded, H] BFloat16
 */
at::Tensor prefill_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::vector<int64_t>& q_offsets,
    const std::vector<int64_t>& q_lens);

} // namespace pace

#endif // PACE_PREFILL_ATTENTION_H
