#include "VulkanCommandBuffer.h"
#include <stdexcept>

VulkanCommandBuffer::VulkanCommandBuffer(const VkInstance instance, const VkDevice device, const VkQueue graphicsQueue, const int maxFramesInFlight, const QueueFamilyIndices queueFamilyIndices)
{
    this->instance = instance;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->maxFramesInFlight = maxFramesInFlight;
    this->queueFamilyIndices = queueFamilyIndices;

    this->tempCommandPool = createCommandPool(queueFamilyIndices, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
}

VulkanCommandBuffer::~VulkanCommandBuffer()
{
    Release();
}

void VulkanCommandBuffer::Release()
{
    // CommandPoolを破棄したらコマンドバッファも自動的に破棄される
    for (auto commandPool : commandPools) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }
    commandPools.clear();
}

void VulkanCommandBuffer::TransitionImageLayoutOnce(const VkDevice device, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, tempCommandPool);

	this->TransitionImageLayout(commandBuffer, image, format, oldLayout, newLayout);

    endSingleTimeCommands(device, tempCommandPool, commandBuffer);
}

void VulkanCommandBuffer::TransitionImageLayout(const VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (hasStencilFormat(format)) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

	BarrierInfo barrierInfo = GetBarrierInfo(oldLayout, newLayout);
	barrier.srcAccessMask = barrierInfo.srcAccessMask;
	barrier.dstAccessMask = barrierInfo.dstAccessMask;

    vkCmdPipelineBarrier(commandBuffer,
        barrierInfo.sourceStage, barrierInfo.destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void VulkanCommandBuffer::copyBuffer(const VkDevice device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, tempCommandPool);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(device, tempCommandPool, commandBuffer);
}

void VulkanCommandBuffer::copyBufferToImage(const VkDevice device, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(device, tempCommandPool);
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
        width,
        height,
        1
    };
    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region // バッファから複数のイメージにコピーする場合は配列で指定できる
    );
    endSingleTimeCommands(device, tempCommandPool, commandBuffer);
}

#pragma region コマンドバッファ
VkCommandPool VulkanCommandBuffer::createCommandPool(const QueueFamilyIndices queueFamilyIndices, const VkCommandPoolCreateFlagBits flags)
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = flags;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    commandPools.push_back(commandPool);

    return commandPool;
}

std::vector<VkCommandBuffer> VulkanCommandBuffer::createCommandBuffers(const VkCommandPool commandPool)
{
    std::vector<VkCommandBuffer> commandBuffers(maxFramesInFlight);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = maxFramesInFlight;

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
    return commandBuffers;
}

VkCommandBuffer VulkanCommandBuffer::beginSingleTimeCommands(const VkDevice device, const VkCommandPool commandPool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanCommandBuffer::endSingleTimeCommands(const VkDevice device, const VkCommandPool commandPool, const VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}
#pragma endregion