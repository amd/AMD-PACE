/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <ops/onednn.h>

namespace pace {

engine& cpu_eng() {
  static engine cpu_eng_(engine::kind::cpu, 0);
  return cpu_eng_;
}

} // namespace pace
