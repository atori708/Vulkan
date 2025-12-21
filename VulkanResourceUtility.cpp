#include "VulkanResourceUtility.h"

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

std::vector<uint32_t> readFileAsUint32(const std::string& filename) {
    // Read the file as a vector of char
    std::vector<char> charBuffer = readFile(filename);

    // Ensure the size of the buffer is a multiple of 4 (size of uint32_t)
    if (charBuffer.size() % sizeof(uint32_t) != 0) {
        throw std::runtime_error("File size is not a multiple of 4 bytes, cannot convert to uint32_t.");
    }

    // Convert the char buffer to a uint32_t buffer
    std::vector<uint32_t> uint32Buffer(charBuffer.size() / sizeof(uint32_t));
    memcpy(uint32Buffer.data(), charBuffer.data(), charBuffer.size());

    return uint32Buffer;
}


uint32_t findMemoryType(const VkPhysicalDevice physicalDevice, const uint32_t typeFilter, const VkMemoryPropertyFlags properties)
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

bool hasStencilFormat(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}