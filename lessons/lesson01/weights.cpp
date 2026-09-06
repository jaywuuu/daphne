#include "weights.h"
#include "types.h"

#include <format>
#include <mio/mio.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace daphne {
void weights_registry::register_safetensor_file(
    const std::filesystem::path file) {
  std::error_code ec;
  mio::mmap_source src = mio::make_mmap_source(file.string(), ec);
  if (ec) {
    throw std::runtime_error(
        std::format("Error: Failed to memory map file: {}\n", ec.message()));
  }

  if (src.size() < sizeof(uint64_t)) {
    throw std::runtime_error(std::format(
        "Error: File too small to contain a valid Safetensors header.\n"));
  }

  const uint64_t nbytes = *reinterpret_cast<const uint64_t *>(src.data());
  if (sizeof(uint64_t) + nbytes > src.size()) {
    throw std::runtime_error(std::format("Error: Corrupt Safetensors header "
                                         "size ({} bytes exceeds file size).\n",
                                         nbytes));
  }

  std::string_view json(src.data() + sizeof(uint64_t), nbytes);

  auto parsed = nlohmann::json::parse(json);
  for (auto &[key, value] : parsed.items()) {
    if (key == "__metadata__")
      continue;

    // more error checking?  Nah.
    weight_entry e(key, value["dtype"].get<std::string>(),
                   value["data_offsets"].get<std::vector<uint64_t>>(),
                   value["shape"].get<std::vector<int64_t>>());
    weights_.push_back(std::move(e));
  }

  // Move to vector to keep the memory map open.  Let the destructor
  // handle cleaning up.
  sources_.push_back(std::move(src));
}

void weights_registry::register_safetensor_files(
    const std::vector<std::filesystem::path> &files) {
  for (auto p : files) {
    register_safetensor_file(p);
  }
}

void weights_registry::register_safetensor_files(
    const std::vector<std::string> &files) {
  std::vector<std::filesystem::path> paths(files.begin(), files.end());
  register_safetensor_files(paths);
}

tensor weights_registry::create_tensor(size_t index) const {
  if (index >= weights_.size()) {
    throw std::runtime_error(std::format(
        "Error: index out of bounds.  Got {} where number of entries is {}\n",
        index, weights_.size()));
  }

  const weight_entry &e = weights_[index];

  return tensor(e.name, e.shape, parse_dtype(e.dtype));
}
} // namespace daphne