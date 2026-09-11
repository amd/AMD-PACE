/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef ONEDNN_H
#define ONEDNN_H

#include "dnnl.hpp"
using namespace dnnl;

namespace pace {

engine& cpu_eng();

} // namespace pace

#endif // ONEDNN_H
