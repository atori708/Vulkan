#include "VulkanSwapChain.h"
#include "VulkanDevice.h"

// TODO SwapChainがデプスとかを作ってるのはおかしいので、別のクラスに分ける

VulkanSwapChain::VulkanSwapChain(GLFWwindow* window, VkPhysicalDevice physicalDevice, VkDevice device,  VulkanQueue* queue, VulkanCommandBuffer* commandBuffer, VulkanResources* resources)
{
    this->window = window;
    this->physicalDevice = physicalDevice;
    this->device = device;
    this->commandBuffer = commandBuffer;
    this->queue = queue;
    this->resources = resources;
}

VulkanSwapChain::~VulkanSwapChain()
{
    Cleanup();
}

void VulkanSwapChain::Initialize(VkSurfaceKHR surface, VkSurfaceFormatKHR surfaceFormat, VkRenderPass renderPass)
{
    this->swapChain = createSwapChain(surface, surfaceFormat, swapChainImages);
    this->swapChainImageViews = createSwapChainImageViews(swapChainImages);
    this->depthImageView = createDepthResources(depthImage, depthImageMemory);
    this->swapChainFramebuffers = createSwapChainFramebuffers(renderPass, swapChainImageViews, depthImageView, swapChainExtent);
}

/// <summary>
/// SwapChainを作成する
/// </summary>
VkSwapchainKHR VulkanSwapChain::createSwapChain(VkSurfaceKHR surface, VkSurfaceFormatKHR surfaceFormat, std::vector<VkImage>& swapChainImages) {
    auto swapChainSupport = querySwapChainSupport(surface);
    if(surfaceFormat.format != VK_FORMAT_UNDEFINED) {
        // 指定されたフォーマットがサポートされているか確認
        bool formatSupported = false;
        for (const auto& availableFormat : swapChainSupport.formats) {
            if (availableFormat.format == surfaceFormat.format && availableFormat.colorSpace == surfaceFormat.colorSpace) {
                formatSupported = true;
                break;
            }
        }
        if (!formatSupported) {
            throw std::runtime_error("requested swap chain format is not supported!");
        }
    }
    auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    auto extent = chooseSwapExtent(swapChainSupport.capabilities);
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; // 最小数より1つ多い画像数を要求する

    // 最大数が指定されている場合はそれを超えないようにする
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    auto indices = queue->FindGraphicsQueueFamilies(physicalDevice, surface);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
    createdSwapChainSurfaceFormat = surfaceFormat;

    return swapChain;
}

SwapChainSupportDetails VulkanSwapChain::querySwapChainSupport(VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
    }
    return details;
}

VkSurfaceFormatKHR VulkanSwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR VulkanSwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) { // triple buffering
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

/// <summary>
/// SwapChainのImageViewを作成する
/// </summary>
std::vector<VkImageView> VulkanSwapChain::createSwapChainImageViews(std::vector<VkImage> swapChainImages)
{
    std::vector<VkImageView> swapChainImageViews(swapChainImages.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++)
    {
        swapChainImageViews[i] = resources->createImageView2D(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);

        // TODO 名前つけるようの静的な関数作る
        //VkDebugUtilsObjectNameInfoEXT nameInfo{};
        //nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        //nameInfo.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
        //nameInfo.objectHandle = (uint64_t)swapChainImageViews[i];
        //nameInfo.pObjectName = "SwapChainImageView " + i;
        //CreateDebugUtilsObjectNameEXT(device, &nameInfo);
    }

    return swapChainImageViews;
}

void VulkanSwapChain::recreateSwapChain(
    const VkSurfaceKHR surface,
    const VkRenderPass renderPass,
    const VkExtent2D swapChainExtent)
{
    // ウィンドウが最小化されたときに幅と高さが0になるので、その場合はリサイズされるまで待機する
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    Cleanup();

    this->swapChain = createSwapChain(surface, createdSwapChainSurfaceFormat, swapChainImages);
    this->swapChainImageViews = createSwapChainImageViews(swapChainImages);
    this->depthImageView = createDepthResources(depthImage, depthImageMemory);
    this->swapChainFramebuffers = createSwapChainFramebuffers(renderPass, swapChainImageViews, depthImageView, swapChainExtent);
}

VkImageView VulkanSwapChain::createDepthResources(VkImage& depthImage, VkDeviceMemory& depthImageMemory)
{
    VkFormat depthFormat = findDepthFormat();
    depthImage = resources->createImage(physicalDevice,
        device,
        swapChainExtent.width, swapChainExtent.height,
        depthFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        depthImageMemory);

    depthImageView = resources->createImageView2D(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    commandBuffer->transitionImageLayout(device, depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    return depthImageView;
}

VkFormat VulkanSwapChain::findDepthFormat()
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
VkFormat VulkanSwapChain::findSupportedFormat(const VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
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

std::vector<VkFramebuffer> VulkanSwapChain::createSwapChainFramebuffers(const VkRenderPass renderPass, const std::vector<VkImageView> swapChainImageViews, const VkImageView depthImageView, const VkExtent2D swapChainExtent)
{
    std::vector<VkFramebuffer> swapChainFramebuffers(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        std::array<VkImageView, 2> attachments = {
            swapChainImageViews[i],
            depthImageView,
        };
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;
        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    return swapChainFramebuffers;
}

void VulkanSwapChain::Cleanup()
{
    if (swapChain == VK_NULL_HANDLE) {
        return;
    }

    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    // ImageViewはSwapChainと一緒に破棄されるので破棄は不要
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    swapChain = VK_NULL_HANDLE;

    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);
}
