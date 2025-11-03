#include "VulkanBufferCreator.h"

/// <summary>
  /// 頂点バッファを作成する
  /// </summary>
VkBuffer VulkanBufferCreator::createVertexBuffer(const std::vector<Vertex> vertices, VkDeviceMemory& bufferMemory)
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    stagingBuffer = CreateBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBufferMemory);

    // Staging Bufferにデータをコピーする
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // GPUのVertex Bufferにデータをコピーする
    VkBuffer vertexAndIndexBuffer;
    vertexAndIndexBuffer = CreateBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        bufferMemory);
    commandBuffer->copyBuffer(device, stagingBuffer, vertexAndIndexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    return vertexAndIndexBuffer;
}

/// <summary>
/// インデックスバッファを作成する
/// </summary>
VkBuffer VulkanBufferCreator::createIndexBuffer(const std::vector<uint16_t> indices, VkDeviceMemory& bufferMemory)
{
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    stagingBuffer = CreateBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBufferMemory);

    // Staging Bufferにデータをコピーする
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // GPUのVertex Bufferにデータをコピーする
    VkBuffer vertexAndIndexBuffer;
    vertexAndIndexBuffer = CreateBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        bufferMemory);
    commandBuffer->copyBuffer(device, stagingBuffer, vertexAndIndexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    return vertexAndIndexBuffer;
}

/// <summary>
/// 頂点バッファとインデックスバッファをまとめて作成する
/// </summary>
VkBuffer VulkanBufferCreator::createVertexAndIndexBuffer(const std::vector<Vertex> vertices, const std::vector<uint16_t> indices, VkDeviceSize& indexBufferOffset, VkDeviceMemory& bufferMemory)
{
    VkDeviceSize vertexBufferSize = sizeof(vertices[0]) * vertices.size();
    VkDeviceSize indexBufferSize = sizeof(indices[0]) * indices.size();
    VkDeviceSize bufferSize = vertexBufferSize + indexBufferSize;
    indexBufferOffset = alignIndexBufferOffset(vertexBufferSize, physicalDevice);

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    stagingBuffer = CreateBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBufferMemory);

    // Staging Bufferにデータをコピーする
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)vertexBufferSize);
    memcpy((char*)data + indexBufferOffset, indices.data(), (size_t)indexBufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // GPUのVertex Bufferにデータをコピーする
    VkBuffer vertexAndIndexBuffer;
    vertexAndIndexBuffer = CreateBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        bufferMemory);
    commandBuffer->copyBuffer(device, stagingBuffer, vertexAndIndexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    return vertexAndIndexBuffer;
}

std::vector<VkBuffer> VulkanBufferCreator::createUniformBuffers(const VkDeviceSize bufferSize, const size_t bufferCount, std::vector<VkDeviceMemory>& uniformBufferMemories, std::vector<void*>& bufferMapped)
{
    std::vector<VkBuffer> uniformBuffers(bufferCount);
    uniformBufferMemories.resize(bufferCount);
    bufferMapped.resize(bufferCount);

    for (size_t i = 0; i < bufferCount; i++) {
        uniformBuffers[i] = CreateBuffer(
            bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBufferMemories[i]);

        vkMapMemory(device, uniformBufferMemories[i], 0, bufferSize, 0, &bufferMapped[i]);
    }

    return uniformBuffers;
}


VkBuffer VulkanBufferCreator::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);

    return buffer;
}

uint32_t VulkanBufferCreator::findMemoryType(const uint32_t typeFilter, const VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

// より適切なアライメントを取る例
VkDeviceSize VulkanBufferCreator::alignIndexBufferOffset(VkDeviceSize offset, VkPhysicalDevice physicalDevice)
{
    // デバイスの制約を取得
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

    // minUniformBufferOffsetAlignment などの値を使用してアライメント
    VkDeviceSize alignment = deviceProperties.limits.minStorageBufferOffsetAlignment;

    // オフセットを適切なアライメント境界に合わせる
    return (offset + alignment - 1) & ~(alignment - 1);
}