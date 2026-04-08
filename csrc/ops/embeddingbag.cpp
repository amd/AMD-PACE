/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 *******************************************************************************/

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <torch/library.h>

#include <core/logging.h>
#include <ops/embeddingbag.h>
#include <ops/kernels/embeddingbag_kernel.h>
#include <utils/utils.h>

namespace pace {

at::Tensor qmerged_embedding_bag_nbit_cat(
    const c10::ArrayRef<c10::intrusive_ptr<EmbeddingPackedParamsBase>> weights,
    const at::TensorList& indices,
    const at::TensorList& offsets,
    const at::Tensor& dense,
    const int64_t bit_width) {
  PROFILE_PACE_FUNCTION("qmerged_embedding_bag_nbit_cat");

  TORCH_CHECK(
      dense.scalar_type() == at::kFloat,
      "Expected Float dense input, but found: ",
      dense.scalar_type());
  TORCH_CHECK(dense.is_contiguous(), "Expected dense input to be contiguous");
  TORCH_CHECK(dense.dim() == 2, "dense input must be 2-dimensional");
  TORCH_CHECK(
      offsets.size() == indices.size(),
      "Number of elements in offsets should be equal to the number of indices, but got ",
      offsets.size(),
      " and ",
      indices.size());

  for (int idx = 0; idx < indices.size(); idx++) {
    TORCH_CHECK(
        (offsets[idx].sizes()[0] - 1) == dense.sizes()[0],
        "Number of elements in offsets should be equal to the batch size + 1, but got ",
        offsets[idx].sizes()[0],
        " and ",
        dense.sizes()[0],
        " at index ",
        idx);
  }

  int num_embedding_bags = weights.size();
  int batch_size = dense.sizes()[0];
  int output_size = dense.sizes()[1];
  int total_output_size = output_size * (num_embedding_bags + /*dense*/ 1);

  at::Tensor output = at::empty({batch_size, total_output_size}, at::kFloat);
  output.index_put_(
      {at::indexing::Slice(), at::indexing::Slice(0, output_size)}, dense);

  for (int idx = 0; idx < num_embedding_bags; idx++) {
    PackedEmbeddingBagWeight* emb_weight =
        static_cast<PackedEmbeddingBagWeight*>(&(*(weights[idx])));
    kernels::qembedding_bag_nbit_with_stride(
        output,
        emb_weight->packed_w,
        indices[idx],
        offsets[idx],
        static_cast<int>(bit_width),
        num_embedding_bags,
        output_size,
        idx);
  }

  // Currently weights are not used in logging since it is packed.
  PROFILE_ADD_INFO_EMBEDDING(
      indices[0], offsets[0], output, {dense}, {"concat"});

  return output;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "qmerged_embedding_bag_nbit_cat(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase[] weights, Tensor[] indices, Tensor[] offsets, Tensor dense, int bit_width) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, CPU, m) {
  m.impl(
      "qmerged_embedding_bag_nbit_cat", pace::qmerged_embedding_bag_nbit_cat);
}

} // namespace
