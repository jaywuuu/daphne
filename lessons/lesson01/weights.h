#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <mio/mio.hpp>

namespace daphne {
class tensor;

struct weight_entry {
  // probably should be immutable but whatever.
  uint64_t size = 0;
  uint64_t start = 0;
  uint64_t end = 0;
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;

  weight_entry(const std::string &name, const std::string &dtype,
               const std::vector<uint64_t> &offsets,
               const std::vector<int64_t> &shape)
      : name(name), dtype(dtype), start(offsets[0]), end(offsets[1]),
        shape(shape) {
    size = end - start;
  }
};

class weights_registry {
private:
  std::vector<weight_entry> weights_;
  std::vector<mio::mmap_source> sources_;

public:
  weights_registry() = default;

  void register_safetensor_file(const std::filesystem::path file);
  void
  register_safetensor_files(const std::vector<std::filesystem::path> &files);
  void register_safetensor_files(const std::vector<std::string> &files);

  size_t size() const noexcept { return weights_.size(); }

  // tensor factory.  maybe not the best place to put this but whatever.
  tensor create_tensor(size_t index) const;
};
} // namespace daphne