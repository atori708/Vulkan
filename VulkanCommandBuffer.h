#pragma once
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include "VulkanDefine.h"


class VulkanCommandBuffer
{
private:
    VkInstance instance;
    VkDevice device;
    VkQueue graphicsQueue;
    int maxFramesInFlight;
    QueueFamilyIndices queueFamilyIndices;

    std::vector<VkCommandPool> commandPools;
    VkCommandPool tempCommandPool;

public:
    VulkanCommandBuffer(const VkInstance instance, const VkDevice device, const VkQueue graphicsQueue, const int maxFramesInFlight, const QueueFamilyIndices queueFamilyIndices);
    ~VulkanCommandBuffer();

    VkCommandPool createCommandPool(const QueueFamilyIndices queueFamilyIndices, const VkCommandPoolCreateFlagBits flags);
    std::vector<VkCommandBuffer> createCommandBuffers(const VkCommandPool commandPool);

    void Release();

    void transitionImageLayout(const VkDevice device, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBuffer(const VkDevice device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void copyBufferToImage(const VkDevice device, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    VkCommandBuffer beginSingleTimeCommands(const VkDevice device, const VkCommandPool commandPool);
    void endSingleTimeCommands(const VkDevice device, const VkCommandPool commandPool, const VkCommandBuffer commandBuffer);
};

