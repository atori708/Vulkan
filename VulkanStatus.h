#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>

class VulkanStatus
{
private:
	int maxAnisotropy = 0;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

public:
    VulkanStatus(VkPhysicalDevice physicalDevice) : physicalDevice(physicalDevice)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	}

	int GetMaxAnisotropy() const { return maxAnisotropy; }

    VkFormat FindDepthFormat()
    {
        return findSupportedFormat(
            physicalDevice,
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    /// <summary>
   /// サポートしているフォーマットを<paramref name="candidates"/>の中から探す
   /// </summary>
    VkFormat findSupportedFormat(const VkPhysicalDevice physicalDevice, const std::vector<VkFormat> candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }
};

