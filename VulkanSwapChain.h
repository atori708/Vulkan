#pragma once
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vector>

#include "VulkanDefine.h"
#include "VulkanQueue.h"
#include "VulkanCommandBuffer.h"
#include "VulkanResources.h"

class VulkanSwapChain
{
private:
    GLFWwindow* window;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkSwapchainKHR swapChain;
    VulkanResources* resources; // スマートぽいんたでもいいのでは？
    VulkanQueue* queue;
    VulkanCommandBuffer* commandBuffer;

    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkSurfaceFormatKHR createdSwapChainSurfaceFormat;

    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkImage depthImage;
    VkImageView depthImageView;
    VkDeviceMemory depthImageMemory;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    SwapChainSupportDetails querySwapChainSupport(VkSurfaceKHR surface);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    std::vector<VkImageView> createSwapChainImageViews(std::vector<VkImage> swapChainImages);


    std::vector<VkFramebuffer> createSwapChainFramebuffers(const VkRenderPass renderPass, const std::vector<VkImageView> swapChainImageViews, const VkImageView depthImageView, const VkExtent2D swapChainExtent);

    VkFormat findDepthFormat() const;
    VkFormat findSupportedFormat(const VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)const;

public:
    VulkanSwapChain(GLFWwindow* window, VkPhysicalDevice physicalDevice, VkDevice device, VulkanQueue* queue, VulkanCommandBuffer* commandBuffer, VulkanResources* resources);
    ~VulkanSwapChain();

    void Initialize(VkSurfaceKHR surface, VkSurfaceFormatKHR surfaceFormat, VkRenderPass renderPass);
    void Cleanup();

    VkSwapchainKHR createSwapChain(VkSurfaceKHR surface, VkSurfaceFormatKHR surfaceFormat, std::vector<VkImage>& swapChainImages);
    void recreateSwapChain(const VkSurfaceKHR surface, const VkRenderPass renderPass, const VkExtent2D swapChainExtent);
    VkImageView createDepthResources(VkImage& depthImage, VkDeviceMemory& depthImageMemory);


    VkSwapchainKHR GetSwapChain() const { return swapChain; }
    VkExtent2D GetSwapChainExtent() const { return swapChainExtent; }

    VkFramebuffer GetFrameBuffer(int index) const { return swapChainFramebuffers[index]; }

	VkImage GetColorImage(int index) const { return swapChainImages[index]; }
	VkImageView GetColorImageView(int index) const { return swapChainImageViews[index]; }
	VkImageView GetDepthImageView() const { return depthImageView; }

    const VkFormat* GetColorFormat() const { return &swapChainImageFormat; }
    VkFormat GetDepthFormat() const { return findDepthFormat(); }
};

