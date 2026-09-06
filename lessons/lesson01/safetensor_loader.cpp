
#include <filesystem>
#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

#include "weights.h"

using namespace daphne;

ABSL_FLAG(std::vector<std::string>, input_files, {},
          "Comma-separated list of input safetensor file paths");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage(
      "Load weights into a registry from safetensor files.\n"
      "Usage:\n"
      "  safetensor_loader "
      "--input_files=<path/to/safetensors>,<path/to/safentesors>...");

  absl::ParseCommandLine(argc, argv);
  std::vector<std::string> files = absl::GetFlag(FLAGS_input_files);

  if (files.size() == 0) {
    std::cerr << "Error: No safetensors file(s) specified.\n\n"
              << absl::ProgramUsageMessage() << std::endl;
    return 1;
  }

  // check paths are valid.
  for (auto &file : files) {
    std::filesystem::path p = file;
    if (!std::filesystem::exists(p)) {
      std::cerr << "Error: Invalid safetensor path: " << p.string()
                << std::endl;
      return 1;
    }
  }

  weights_registry weights;

  // register weights from each safetensor file in files.
  weights.register_safetensor_files(files);
  std::cout << "Loaded " << weights.size() << " weights from " << files.size()
            << " files\n";

  return 0;
}