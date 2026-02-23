#pragma once
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <array>
#include <chrono>

#include "VulkanDefine.h"
#include "VulkanQueue.h"

class VulkanContext
{
private:

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice;
    VulkanQueue queue;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    float deviceMaxAnisotropy = 1;

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

public:
    VkInstance GetInstance() { return instance; }
    VkPhysicalDevice GetPhysicalDevice() { return physicalDevice; }
    VkDevice GetLogicalDevice() { return logicalDevice; }
    VkQueue GetGraphicsQueue() { return graphicsQueue; }
    VkQueue GetPresentQueue() { return presentQueue; }
    VkSurfaceKHR GetSurface() { return surface; }
    float GetDeviceMaxAnisotropy() { return deviceMaxAnisotropy; }

public:
    void InitializeVulkan(GLFWwindow* window, VulkanQueue& queue);
    void CleanupVulkan();


private:
    VkInstance CreateInstance();
    VkSurfaceKHR createSurface(GLFWwindow* window);

    std::vector<const char*> getRequiredExtensions();

#pragma region Physical Device
    VkPhysicalDevice pickPhysicalDevice(VkInstance vkInstance);
    bool isDeviceSuitable(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    int rateDeviceSuitability(VkPhysicalDevice device);
#pragma endregion

#pragma region Logical Device
    VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, QueueFamilyIndices queueFamilyIndices);
#pragma endregion



#pragma region Validation Layers
    bool checkValidationLayerSupport();
    void setupDebugMessenger(VkInstance vkInstance);
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    VkResult CreateDebugUtilsObjectNameEXT(VkDevice device, const VkDebugUtilsObjectNameInfoEXT* pNameInfo);
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
#pragma endregion


};

