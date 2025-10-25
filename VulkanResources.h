#pragma once
#include <vulkan/vulkan.h>

#include <iostream>
#include <fstream>
#include <vector>

#include "VulkanDefine.h"
#include "VulkanResourceUtility.h"

class VulkanResources
{
private:
    VkDevice device;

public:
    VulkanResources(VkDevice device);

#pragma region ShaderModule
    VkShaderModule createShaderModule(const std::string& filename);
    VkShaderModule createShaderModuleFromBinary(const std::vector<char>& code);
#pragma endregion

    VkImage createImage(const VkPhysicalDevice physicalDevice, const VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory& imageMemory);
    VkImageView createImageView2D(const VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
};

