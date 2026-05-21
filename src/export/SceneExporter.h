#pragma once
#include <string>

namespace anacapa {

// Read the binary scene blob at blobPath and write a USD scene to usdPath.
// The materials/ directory (containing .mtlx/.osl files written by the Python
// addon) must exist at the same directory level as usdPath.
// Returns true on success.
bool exportSceneToUsd(const std::string& blobPath, const std::string& usdPath);

} // namespace anacapa
