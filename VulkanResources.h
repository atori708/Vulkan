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
    VulkanResources(const VkDevice device);

#pragma region ShaderModule
    VkShaderModule createShaderModule(const std::string& filename);
    VkShaderModule createShaderModuleFromBinary(const std::vector<char>& code);
#pragma endregion

    VkImage createImage(const VkPhysicalDevice physicalDevice, const VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory& imageMemory);
    VkImageView createImageView2D(const VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

    const std::vector<VkDescriptorSet> createDescriptorSets(const VkDevice device, const VkDescriptorPool descriptorPool, const VkDescriptorSetLayout descriptorSetLayout, const uint32_t descriptorCount) const
    {
        std::vector<VkDescriptorSetLayout> layouts(descriptorCount, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(descriptorCount);
        allocInfo.pSetLayouts = layouts.data();

        std::vector<VkDescriptorSet> modelDescriptorSets(descriptorCount);
        auto result = vkAllocateDescriptorSets(device, &allocInfo, modelDescriptorSets.data());

        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        return modelDescriptorSets;
    }
};

