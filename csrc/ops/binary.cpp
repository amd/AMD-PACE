/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/quantized/Quantizer.h>
#include <core/logging.h>
#include <ops/binary.h>
#include <ops/kernels/binary_kernels.h>
#include <torch/library.h>
#include <utils/utils.h>

namespace pace {

// Wrapper function to support multiple input types
at::Tensor qmul_add(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& addend,
    const at::Scalar& output_scale,
    const at::Scalar& output_zpoint,
    const at::ScalarType& output_dtype) {
  PROFILE_PACE_FUNCTION("qmul_add");

  TORCH_CHECK(
      (a.sizes() == b.sizes() && a.sizes() == addend.sizes()),
      "pace::qmul_add requires all three inputs (a, b, and addend) to "
      "be of the same shape, but recieved: ",
      a.sizes(),
      ", ",
      b.sizes(),
      ", and ",
      addend.sizes());
  TORCH_CHECK(
      a.dim() == 2,
      "pace::qmul_add only supports 2D inputs, but recieved: ",
      a.sizes(),
      ", ",
      b.sizes(),
      ", and ",
      addend.sizes());
  TORCH_CHECK(
      (a.size(a.dim() - 1) % 96 == 0),
      "pace::qmul_add requires 2nd dimension to be a factor of 96.");
  TORCH_CHECK(
      a.scalar_type() == at::kFloat,
      "pace::qmul_add only supports Float Tensor for A input, got ",
      a.scalar_type());
  TORCH_CHECK(
      b.scalar_type() == at::kQUInt8,
      "pace::qmul_add only supports QUInt8 Tensor for B input, got ",
      a.scalar_type());
  TORCH_CHECK(
      dtype_supported(addend.scalar_type(), {at::kQUInt8, at::kFloat}),
      "pace::qmul_add only support QUInt8 and Float Tensor for Addend "
      "input, got ",
      addend.scalar_type());

  // Get output scale and zero point and create output tensor
  int o_zpoint = output_zpoint.toInt();
  float o_scale = output_scale.toFloat();

  at::QuantizerPtr output_quantizer =
      at::make_per_tensor_affine_quantizer(o_scale, o_zpoint, output_dtype);
  auto opt = c10::TensorOptions().dtype(output_dtype).device(at::kCPU);
  at::Tensor output = at::new_qtensor(
      /*sizes=*/a.sizes(), opt, output_quantizer);

  if (addend.scalar_type() == at::kFloat) {
    kernels::qmul_add_kernel<float>(a, b, addend, output);
  } else {
    kernels::qmul_add_kernel<uint8_t>(a, b, addend, output);
  }

  PROFILE_ADD_INFO_BINARY(a, b, output, {addend}, {"add"});
  return output;
}

} // namespace pace

namespace {

TORCH_LIBRARY_FRAGMENT(pace, m) {
  m.def(
      "qmul_add(Tensor a, Tensor b, Tensor addend, Scalar o_scale, Scalar o_zero_point, ScalarType o_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(pace, QuantizedCPU, m) {
  m.impl("qmul_add", pace::qmul_add);
}

} // namespace
