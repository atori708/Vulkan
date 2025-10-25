#pragma once
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include "VulkanDefine.h"
class VulkanQueue
{
public:
    QueueFamilyIndices FindGraphicsQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
};

