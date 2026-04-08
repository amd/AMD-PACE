/******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Inspired by vLLM's scratchpad memory management
 * (https://github.com/vllm-project/vllm)
 ******************************************************************************/

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "scratchpad_manager.h"

PaceScratchPadManager::PaceScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

PaceScratchPadManager::~PaceScratchPadManager() {
  if (ptr_ != nullptr) {
    std::free(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }
}

void PaceScratchPadManager::realloc(size_t new_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  new_size = round(new_size);
  if (new_size > size_) {
    void* new_ptr = std::aligned_alloc(64, new_size);
    if (new_ptr == nullptr) {
      throw std::runtime_error(
          "PaceScratchPadManager: failed to allocate " +
          std::to_string(new_size) + " bytes for attention scratchpad");
    }
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = new_ptr;
    size_ = new_size;
  }
}

PaceScratchPadManager* PaceScratchPadManager::get_scratchpad_manager() {
  static PaceScratchPadManager manager;
  return &manager;
}
