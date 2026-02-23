#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include <stdexcept>

#include "MeshFormat.h"
#include "VulkanCommandBuffer.h"

class VulkanBufferCreator
{
private:
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VulkanCommandBuffer* commandBuffer;

    const uint32_t findMemoryType(const uint32_t typeFilter, const VkMemoryPropertyFlags properties) const;

    VkDeviceSize alignIndexBufferOffset(VkDeviceSize offset, VkPhysicalDevice physicalDevice);

public:
    VulkanBufferCreator(VkPhysicalDevice physicalDevice, VkDevice device, VulkanCommandBuffer* commandBuffer)
        : physicalDevice(physicalDevice), device(device), commandBuffer(commandBuffer) {
    }

    VkBuffer createVertexBuffer(const std::vector<Vertex> vertices, VkDeviceMemory& bufferMemory);
    VkBuffer createIndexBuffer(const std::vector<uint16_t> indices, VkDeviceMemory& bufferMemory);
    VkBuffer createVertexAndIndexBuffer(const std::vector<Vertex> vertices, const std::vector<uint16_t> indices, VkDeviceSize& indexBufferOffset, VkDeviceMemory& bufferMemory);

    const std::vector<VkBuffer> createUniformBuffers(const VkDeviceSize bufferSize, const size_t bufferCount, std::vector<VkDeviceMemory>& uniformBufferMemories, std::vector<void*>& bufferMapped) const;

    const VkBuffer CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory& bufferMemory) const;
};

