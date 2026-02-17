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

    VkImage invalidTextureImage;
    VkDeviceMemory invalidTextureImageMemory;
    VkImageView invalidTextureImageView;
    VkSampler invalidTextureSampler;

public:
    VulkanTextureCreator(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VulkanBufferCreator* bufferCreator,
        VulkanResources* vulkanResources,
        VulkanCommandBuffer* commandBuffer,
        float maxAnisotropy); // TODO Limit的なクラスを作って渡す
    ~VulkanTextureCreator();

    const VkImage createTextureImage(const std::string texturePath, VkDeviceMemory& textureImageMemory)const;
    const VkSampler createTextureSampler(float anisotropy) const;

    const VkImage GetInvalidTextureImage() const { return invalidTextureImage; }
    const VkImageView GetInvalidTextureImageView() const { return invalidTextureImageView; }
    const VkSampler GetInvalidTextureSampler() const { return invalidTextureSampler; }
};

