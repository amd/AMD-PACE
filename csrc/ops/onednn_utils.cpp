/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ops/onednn.h>
#include <ops/onednn_utils.h>

namespace pace {

memory view_tensor_as_memory(
    const at::Tensor& atensor,
    memory::dims dims,
    memory::data_type data_type,
    memory::format_tag format) {
  TORCH_CHECK(
      atensor.is_contiguous(),
      "view_tensor_as_memory: input tensor must be contiguous");
  memory::desc zen_memory_desc = memory::desc(dims, data_type, format);
  return memory({zen_memory_desc}, cpu_eng(), atensor.data_ptr());
}

memory view_tensor_as_memory(
    const at::Tensor& atensor,
    transpose_dims trans_dims,
    int ndims) {
  memory::dims sizes = atensor.sizes().vec();
  if (ndims != 0 && ndims != atensor.dim()) {
    std::vector<int64_t> new_dims(ndims - atensor.dim(), 1);
    new_dims.insert(new_dims.end(), sizes.begin(), sizes.end());
    sizes = new_dims;
  }

  if (trans_dims.dim_a != trans_dims.dim_b) {
    TORCH_CHECK(
        trans_dims.dim_a < static_cast<int>(sizes.size()) &&
            trans_dims.dim_b < static_cast<int>(sizes.size()),
        "trans_dims is out of range");
    std::swap(sizes[trans_dims.dim_a], sizes[trans_dims.dim_b]);
  }

  tag format = get_default_format(sizes.size(), trans_dims);

  return view_tensor_as_memory(
      atensor, sizes, dtype_to_onednn(atensor.scalar_type()), format);
}

at::Tensor view_memory_as_tensor(
    const memory& mem,
    const at::ScalarType& scalar_type,
    transpose_dims trans_dims) {
  auto mem_desc = mem.get_desc();
  auto dims = mem_desc.get_dims();
  std::vector<int64_t> tensor_dims(dims.begin(), dims.end());

  if (tensor_dims.size() == 2 && trans_dims.dim_a != trans_dims.dim_b) {
    TORCH_CHECK(
        tensor_dims.size() >
            static_cast<size_t>(std::max(trans_dims.dim_a, trans_dims.dim_b)),
        "view_memory_as_tensor: trans_dims is out of range for tensor dimensions");
    std::swap(tensor_dims[trans_dims.dim_a], tensor_dims[trans_dims.dim_b]);
  }

  void* data_ptr = mem.get_data_handle();
  auto options = at::TensorOptions().dtype(scalar_type).device(at::kCPU);
  return at::from_blob(data_ptr, tensor_dims, options);
}

} // namespace pace
