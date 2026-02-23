#include "VulkanTextureCreator.h"

VulkanTextureCreator::VulkanTextureCreator(VkDevice device, VkPhysicalDevice physicalDevice, VulkanBufferCreator* bufferCreator, VulkanResources* vulkanResources, VulkanCommandBuffer* commandBuffer, float maxAnisotropy)
{
    this->device = device;
    this->physicalDevice = physicalDevice;
    this->bufferCreator = bufferCreator;
    this->vulkanResources = vulkanResources;
    this->commandBuffer = commandBuffer;
    this->deviceMaxAnisotropy = maxAnisotropy;

	// 無効なテクスチャの作成
    invalidTextureImage = createTextureImage("Assets/Embedded/white.png", invalidTextureImageMemory);
	invalidTextureImageView = vulkanResources->createImageView2D(invalidTextureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
	invalidTextureSampler = createTextureSampler(deviceMaxAnisotropy);
}

VulkanTextureCreator::~VulkanTextureCreator()
{
    vkDestroySampler(device, invalidTextureSampler, nullptr);
    vkDestroyImageView(device, invalidTextureImageView, nullptr);
    vkDestroyImage(device, invalidTextureImage, nullptr);
    vkFreeMemory(device, invalidTextureImageMemory, nullptr);
}

const VkImage VulkanTextureCreator::createTextureImage(const std::string texturePath, VkDeviceMemory& textureImageMemory) const
{
    // 画像読み込み
    int width, height, channels;
    stbi_uc* pixels = stbi_load(texturePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    VkDeviceSize imageSize = width * height * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image! path:" + texturePath);
    }

    // 画像をStagingBufferに転送
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    stagingBuffer = bufferCreator->CreateBuffer(
        imageSize,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);
    stbi_image_free(pixels);

    // VkImageの作成
    VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
    VkImage textureImage = vulkanResources->createImage(physicalDevice, device,
        width, height,
        format,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        textureImageMemory);

    // StagingBufferからVkImageへ転送
    commandBuffer->TransitionImageLayoutOnce(device, textureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    commandBuffer->copyBufferToImage(device, stagingBuffer, textureImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    // シェーダから参照できるようにLayoutを変更
    commandBuffer->TransitionImageLayoutOnce(device, textureImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    return textureImage;
}
#pragma endregion

#pragma region TextureSampler
const VkSampler VulkanTextureCreator::createTextureSampler(float anisotropy = 0)const
{
    if (anisotropy > deviceMaxAnisotropy) {
        anisotropy = deviceMaxAnisotropy;
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = anisotropy > 0;
    samplerInfo.maxAnisotropy = anisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE; // VK_TRUEにすると[0, texWidth)や[0, texHeight)の範囲でサンプリングする まあ使うことはない
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    VkSampler textureSampler;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }

    return textureSampler;
}
#pragma endregion