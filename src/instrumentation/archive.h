#pragma once
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nacelle::profile {
bool validateStoredArchive(std::span<const char> bytes, unsigned minimumEntries,
                           unsigned maximumEntries, std::string& error);
bool readStoredEntry(std::span<const char> bytes, const char* name, size_t limit,
                     std::string& result, std::string& error);
bool writeStoredArchive(const std::string& path,
                        const std::vector<std::pair<std::string, std::string_view>>& entries,
                        std::string& error);
}
