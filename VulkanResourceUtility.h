#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <fstream>
#include <vector>

/// <summary>
/// ファイルを読み込む
/// </summary>
extern std::vector<char> readFile(const std::string& filename);

extern std::vector<uint32_t> readFileAsUint32(const std::string& filename) ;

extern uint32_t findMemoryType(const VkPhysicalDevice physicalDevice, const uint32_t typeFilter, const VkMemoryPropertyFlags properties);

extern bool hasStencilFormat(VkFormat format);
