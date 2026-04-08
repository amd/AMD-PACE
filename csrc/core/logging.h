/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_LOGGING_H
#define PACE_LOGGING_H

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>

#include <ATen/ATen.h>

namespace pace {

namespace logging {

/**
 * @brief Convert the dtype to string for logs
 *
 * @param type
 * @return std::string
 */
std::string dtype_to_string(const at::ScalarType& type);

/**
 * @brief Convert the dtype to string for logs
 *
 * @param atensor
 * @return std::string
 */
std::string dtype_to_string(const at::Tensor& atensor);

/**
 * @brief Convert the sizes to string for logs
 *
 * @param atensor
 * @return std::string
 */
std::string sizes_to_string(const at::IntArrayRef& atensor_size);

/**
 * @brief Convert the sizes to string for logs
 *
 * @param atensor
 * @return std::string
 */
std::string sizes_to_string(at::Tensor atensor);

/**
 * @brief Get the current time in string format
 *
 * @return std::string
 */
std::string get_current_time();

enum PACELogLevel {
  PACELogLevelDebug,
  PACELogLevelProfile,
  PACELogLevelInfo,
  PACELogLevelWarning,
  PACELogLevelError,
  PACENoLog,
};

class StatelessLogger {
 public:
  template <typename... Args>
  static void log(PACELogLevel level, Args... args) {
    if (level < getPACELogLevelThreshold()) {
      return;
    }

    std::ostringstream buffer;
    buffer << get_current_time() << " ";
    buffer << getPACELogLevelName(level) << " ";
    print(buffer, std::forward<Args>(args)...);
    buffer << '\n';

    std::cout << buffer.str();
  }

 private:
  // Variadic template print function
  template <typename T, typename... Args>
  static void print(std::ostringstream& buffer, T t, Args... args) {
    buffer << t;
    print(buffer, std::forward<Args>(args)...);
  }

  static void print(std::ostringstream& buffer) {
    (void)buffer;
  }

  static std::string getPACELogLevelName(PACELogLevel level) {
    switch (level) {
      case PACELogLevelDebug:
        return "D";
      case PACELogLevelProfile:
        return "P";
      case PACELogLevelInfo:
        return "I";
      case PACELogLevelWarning:
        return "W";
      case PACELogLevelError:
        return "E";
      default:
        return "I";
    }
  }

  static PACELogLevel getPACELogLevelThreshold() {
    const char* env_value = std::getenv("PACE_LOG_LEVEL");
    if (env_value) {
      if (std::string(env_value) == "debug") {
        return PACELogLevelDebug;
      } else if (std::string(env_value) == "profile") {
        return PACELogLevelProfile;
      } else if (std::string(env_value) == "info") {
        return PACELogLevelInfo;
      } else if (std::string(env_value) == "warning") {
        return PACELogLevelWarning;
      } else if (std::string(env_value) == "error") {
        return PACELogLevelError;
      } else if (std::string(env_value) == "none") {
        return PACENoLog;
      }
    }
    return PACELogLevelInfo;
  }
};

#define PACE_LOG_DEBUG(...) StatelessLogger::log(PACELogLevelDebug, __VA_ARGS__)
#define PACE_LOG_PROFILE(...) \
  StatelessLogger::log(PACELogLevelProfile, __VA_ARGS__)
#define PACE_LOG_INFO(...) StatelessLogger::log(PACELogLevelInfo, __VA_ARGS__)
#define PACE_LOG_WARNING(...) \
  StatelessLogger::log(PACELogLevelWarning, __VA_ARGS__)
#define PACE_LOG_ERROR(...) StatelessLogger::log(PACELogLevelError, __VA_ARGS__)

#define PACE_LOG_CUSTOM(level, ...) StatelessLogger::log(level, __VA_ARGS__)

class TimingLogger {
 public:
  TimingLogger(std::string name, std::string file) : name_(std::move(name)) {
    start_time_ = std::chrono::high_resolution_clock::now();
    std::filesystem::path filePath(file);
    file_ = filePath.filename().string().c_str();
  }

  template <typename... Ts>
  void AddInfo(Ts... vs) {
    std::lock_guard<std::mutex> lock(info_mutex_);
    std::stringstream ss;
    (ss << ... << vs);
    info_ += ss.str();
  }

  void AddInfo(const std::string& info) {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_ += std::move(info);
  }

  void AddInfoBinary(
      const at::Tensor& a,
      const at::Tensor& b,
      const at::Tensor& output,
      const at::TensorList& post_ops,
      const std::vector<std::string>& post_ops_algo);
  void AddInfoLinear(
      const at::Tensor& input,
      const at::Tensor& weight,
      const at::Tensor& bias,
      const at::Tensor& output,
      const at::TensorList& post_ops,
      const std::vector<std::string>& post_ops_algo);
  void AddInfoEmbedding(
      const at::Tensor& indices,
      const at::Tensor& offsets,
      const at::Tensor& output,
      const at::TensorList& post_ops,
      const std::vector<std::string>& post_ops_algo);
  void AddInfoMlpMlpFusion(
      const at::Tensor& src,
      const std::vector<at::Tensor>& weights1,
      const c10::optional<std::vector<at::Tensor>>& bias1,
      const std::vector<at::Tensor>& weights2,
      const c10::optional<at::Tensor>& bias2,
      std::string nlf,
      const c10::optional<std::vector<at::Tensor>>& weights_gateProj,
      const c10::optional<std::vector<at::Tensor>>& bias_gateProj,
      const at::Tensor& output);

  ~TimingLogger() {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time_;
    PACE_LOG_PROFILE(
        file_, " pace::", name_, ",", info_, ",", duration.count());
  }

 private:
  std::mutex info_mutex_;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::string name_;
  std::string info_;
  std::string file_;
};

} // namespace logging

#define PROFILE_PACE_FUNCTION(name) logging::TimingLogger timer(name, __FILE__);
#define PROFILE_ADD_INFO(...) timer.AddInfo(__VA_ARGS__);
#define PROFILE_ADD_INFO_BINARY(a, b, output, post_ops, post_ops_algo) \
  timer.AddInfoBinary(a, b, output, post_ops, post_ops_algo);
#define PROFILE_ADD_INFO_LINEAR(                          \
    input, weight, bias, output, post_ops, post_ops_algo) \
  timer.AddInfoLinear(input, weight, bias, output, post_ops, post_ops_algo);
#define PROFILE_ADD_INFO_EMBEDDING(                    \
    indices, offsets, output, post_ops, post_ops_algo) \
  timer.AddInfoEmbedding(indices, offsets, output, post_ops, post_ops_algo);
#define PROFILE_ADD_INFO_MLP_MLP_FUSION( \
    src,                                 \
    weights1,                            \
    bias1,                               \
    weights2,                            \
    bias2,                               \
    nlf,                                 \
    weights_gateProj,                    \
    bias_gateProj,                       \
    output)                              \
  timer.AddInfoMlpMlpFusion(             \
      src,                               \
      weights1,                          \
      bias1,                             \
      weights2,                          \
      bias2,                             \
      nlf,                               \
      weights_gateProj,                  \
      bias_gateProj,                     \
      output);
// External method to be called from python
inline void pace_logger(int64_t logLevel, std::string message) {
  logging::PACE_LOG_CUSTOM(
      static_cast<logging::PACELogLevel>(logLevel), message);
}

} // namespace pace

#endif // PACE_LOGGING_H
