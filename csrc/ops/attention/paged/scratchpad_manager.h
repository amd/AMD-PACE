/******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Inspired by vLLM's scratchpad memory management
 * (https://github.com/vllm-project/vllm)
 ******************************************************************************/

#ifndef SCRATCHPAD_MANAGER_H
#define SCRATCHPAD_MANAGER_H

#include <cstddef>
#include <cstdio>
#include <mutex>

class PaceScratchPadManager {
 public:
  static constexpr size_t allocation_unit = 4 * 1024; // 4KB

  static PaceScratchPadManager* get_scratchpad_manager();

  PaceScratchPadManager();
  ~PaceScratchPadManager();

  PaceScratchPadManager(const PaceScratchPadManager&) = delete;
  PaceScratchPadManager& operator=(const PaceScratchPadManager&) = delete;

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(ptr_);
  }

  static size_t round(size_t size) {
    return ((size + allocation_unit - 1) / allocation_unit) * allocation_unit;
  }

  void realloc(size_t new_size);

 private:
  std::mutex mutex_;
  size_t size_;
  void* ptr_;
};

#endif
