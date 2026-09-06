
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <mio/mio.hpp>
#include <nlohmann/json.hpp>

ABSL_FLAG(std::string, file, "", "Path to the .safetensors file to inspect");
ABSL_FLAG(int, indent, 2, "JSON indentation spaces (-1 for compact)");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage(
      "Inspects and dumps the JSON metadata header of a Safetensors file.\n"
      "Usage:\n"
      "  safetensor_dmp <path/to/safetensors>\n"
      "  safetensor_dmp --file=<path/to/safetensors>");

  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);

  std::filesystem::path safetensor_path = absl::GetFlag(FLAGS_file);
  if (safetensor_path.empty() && positional_args.size() > 1) {
    safetensor_path = positional_args[1];
  }

  if (safetensor_path.empty()) {
    std::cerr << "Error: No safetensors file specified.\n\n"
              << absl::ProgramUsageMessage() << std::endl;
    return 1;
  }

  if (!std::filesystem::exists(safetensor_path)) {
    std::cerr << "Error: Invalid safetensor path: " << safetensor_path.string()
              << std::endl;
    return 1;
  }

  std::error_code ec;
  mio::mmap_source src = mio::make_mmap_source(safetensor_path.string(), ec);
  if (ec) {
    std::cerr << "Error: Failed to memory map file: " << ec.message()
              << std::endl;
    return 1;
  }

  if (src.size() < sizeof(uint64_t)) {
    std::cerr << "Error: File too small to contain a valid Safetensors header."
              << std::endl;
    return 1;
  }

  const uint64_t nbytes = *reinterpret_cast<const uint64_t *>(src.data());
  if (sizeof(uint64_t) + nbytes > src.size()) {
    std::cerr << "Error: Corrupt Safetensors header size (" << nbytes
              << " bytes exceeds file size)." << std::endl;
    return 1;
  }

  std::string_view json(src.data() + sizeof(uint64_t), nbytes);

  try {
    auto parsed = nlohmann::json::parse(json);
    const int indent = absl::GetFlag(FLAGS_indent);
    std::cout << parsed.dump(indent) << std::endl;
  } catch (const nlohmann::json::exception &e) {
    std::cerr << "Failed to parse JSON header: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}