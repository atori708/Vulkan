#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <set>

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool HasGraphicsQueueFamily() {
        return graphicsFamily.has_value();
    }

    bool HasPresentQueueFamily() {
        return presentFamily.has_value();
    }

    bool IsComplete() {
        return HasGraphicsQueueFamily() && HasPresentQueueFamily();
    }

    std::set<uint32_t> UniqueFamilies() {
        return { graphicsFamily.value(), presentFamily.value() };
    }
};