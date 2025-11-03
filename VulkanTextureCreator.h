#pragma once
#include <vulkan/vulkan.h>
#include <stb_image.h>
#include <stdexcept>

#include "VulkanBufferCreator.h"
#include "VulkanResources.h"
#include "VulkanCommandBuffer.h"

class VulkanTextureCreator
{
private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VulkanBufferCreator* bufferCreator;
    VulkanResources* vulkanResources;
    VulkanCommandBuffer* commandBuffer;
    float deviceMaxAnisotropy = 1;

public:
    VulkanTextureCreator(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VulkanBufferCreator* bufferCreator,
        VulkanResources* vulkanResources,
        VulkanCommandBuffer* commandBuffer,
        float maxAnisotropy); // TODO Limit的なクラスを作って渡す

    VkImage createTextureImage(const std::string texturePath, VkDeviceMemory& textureImageMemory);
    VkSampler createTextureSampler(float anisotropy);
};

