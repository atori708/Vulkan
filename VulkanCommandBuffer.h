#pragma once
#include "VulkanDefine.h"
#include "VulkanResourceUtility.h"

struct BarrierInfo {
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
};

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

    BarrierInfo GetBarrierInfo(VkImageLayout oldLayout, VkImageLayout newLayout) const
    {
		BarrierInfo barrierInfo = {};

        // 転送元・転送先のアクセスを設定(TODO よくわからんので、TutorialのImagesの項目参照
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrierInfo.srcAccessMask = 0;
            barrierInfo.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            barrierInfo.sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            barrierInfo.destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrierInfo.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrierInfo.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            barrierInfo.sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            barrierInfo.destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrierInfo.srcAccessMask = 0;
            barrierInfo.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            barrierInfo.sourceStage =VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            barrierInfo.destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            barrierInfo.srcAccessMask = 0;
            barrierInfo.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            barrierInfo.sourceStage =VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            barrierInfo.destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            barrierInfo.srcAccessMask = 0;
            barrierInfo.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            barrierInfo.sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            barrierInfo.destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
            barrierInfo.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrierInfo.dstAccessMask = 0;
            barrierInfo.sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrierInfo.destinationStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

		return barrierInfo;
    }

    void TransitionImageLayoutOnce(const VkDevice device, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void TransitionImageLayout(const VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBuffer(const VkDevice device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void copyBufferToImage(const VkDevice device, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    VkCommandBuffer beginSingleTimeCommands(const VkDevice device, const VkCommandPool commandPool);
    void endSingleTimeCommands(const VkDevice device, const VkCommandPool commandPool, const VkCommandBuffer commandBuffer);
};

