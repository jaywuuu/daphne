#pragma once

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace daphne {

using shape = std::vector<int64_t>;
using strides = std::vector<int64_t>;

enum class dtype : int { unknown = 0, int4, int8, bf16, fp32 };

constexpr dtype parse_dtype(std::string_view str) {
  if (str == "INT4" || str == "int4")
    return dtype::int4;
  if (str == "INT8" || str == "int8")
    return dtype::int8;
  if (str == "BF16" || str == "bf16")
    return dtype::bf16;
  if (str == "F32" || str == "fp32" || str == "F16")
    return dtype::fp32;
  return dtype::unknown;
}

constexpr size_t element_size(dtype dt) {
  switch (dt) {
  case dtype::int4:
    return 1;
  case dtype::int8:
    return 1;
  case dtype::bf16:
    return 2;
  case dtype::fp32:
    return 4;
  default:
    return 0;
  }
}

struct bf16 {
  uint16_t bits = 0;

  bf16() = default;
  constexpr explicit bf16(uint16_t raw_bits) : bits(raw_bits) {}

  // Convert float -> bf16 (with rounding)
  explicit bf16(float f) {
    uint32_t u = std::bit_cast<uint32_t>(f);
    // Round to nearest even before truncating
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
    bits = static_cast<uint16_t>((u + rounding_bias) >> 16);
  }

  // Convert bf16 -> float (trivial 16-bit left shift)
  constexpr operator float() const {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    return std::bit_cast<float>(u);
  }
};

// Ensure it has the exact same layout and size as uint16_t
static_assert(sizeof(bf16) == 2);
static_assert(alignof(bf16) == 2);

class tensor {
private:
  std::string name_;
  shape shape_;
  strides strides_;
  dtype dtype_;

  // not sure about the void *data, but whatever for now.
  const std::byte *data_ = nullptr;

  static strides compute_strides(const shape &s) {
    if (s.empty())
      return {};
    daphne::strides st(s.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(s.size()) - 1; i >= 0; --i) {
      st[i] = stride;
      stride *= s[i];
    }
    return st;
  }

public:
  tensor(const std::string_view name, const shape &s, dtype dty,
         const void *data = nullptr)
      : name_(name), shape_(s), strides_(compute_strides(s)), dtype_(dty),
        data_(static_cast<const std::byte *>(data)) {}

  const shape &shape() const { return shape_; }
  const strides &strides() const { return strides_; }
  size_t rank() const { return shape_.size(); }
  dtype dtype() const { return dtype_; }
  const std::string &name() const { return name_; }
  const void *raw_data() const { return data_; }

  int64_t numel() const {
    if (shape_.empty())
      return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL,
                           std::multiplies<int64_t>());
  }

  template <std::integral... Indices>
  int64_t linear_index(Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "Must provide at least one index");
    assert(sizeof...(indices) == shape_.size() &&
           "Index count must match tensor rank");

    const int64_t idxs[] = {static_cast<int64_t>(indices)...};
    int64_t flat_idx = 0;
    for (size_t i = 0; i < sizeof...(indices); ++i) {
      assert(idxs[i] >= 0 && idxs[i] < shape_[i] && "Index out of bounds");
      flat_idx += idxs[i] * strides_[i];
    }
    return flat_idx;
  }

  // C++23 multi-dimensional subscript: tensor[a, b, c]
  // My favourite feature of C++23.
  template <std::integral... Indices>
  const std::byte *operator[](Indices... indices) const {
    assert(data_ != nullptr && "Tensor data pointer is null");
    int64_t idx = linear_index(indices...);
    return data_ + (idx * element_size(dtype_));
  }

  // Typed accessor: tensor.at<float>(a, b, c)
  template <typename T, std::integral... Indices>
  const T &at(Indices... indices) const {
    return *reinterpret_cast<const T *>(operator[](indices...));
  }
};
} // namespace daphne