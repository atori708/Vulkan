#include "Material.h"

Material::Material(const VkDevice device, const VulkanBufferCreator* bufferCreator, const VulkanTextureCreator* textureCreator, const VulkanResources* vulkanResources, VkDescriptorPool descriptorPool, Shader* shader, int count)
    : device(device), count(count), shader(shader), bufferInfo(shader->GetBufferInfoBySet(1))
{
    materialUniformBuffers = bufferCreator->createUniformBuffers(bufferInfo.bufferSize, count, materialUniformBufferMemories, materialUniformBuffersMapped);
    materialDescriptorSets = vulkanResources->createDescriptorSets(device, descriptorPool, shader->MaterialDescriptorSetLayout(), count);

    // 最初は無効テクスチャを割り当てておく
    textureImageView = textureCreator->GetInvalidTextureImageView();
    textureSampler = textureCreator->GetInvalidTextureSampler();
    updateDescriptorSets(device);

    _matrixMap["model"] = glm::mat4x4(1.0f);
}

void Material::Release(VkDevice device)
{
    for (int i = 0; i < count; i++) {
        vkDestroyBuffer(device, materialUniformBuffers[i], nullptr);
        vkFreeMemory(device, materialUniformBufferMemories[i], nullptr);
    }
}

void Material::SetFloat(const std::string& name, const float value)
{
    _floatMap[name] = value;
}

void Material::SetVector3(const std::string& name, const glm::vec3 vec3)
{
    _vec3Map[name] = vec3;
}

void Material::SetMatrix4x4(const std::string& name, const glm::mat4x4 matrix)
{
    _matrixMap[name] = matrix;
}

void Material::SetTexture(VkImageView imageView, VkSampler sampler)
{
    textureImageView = imageView;
    textureSampler = sampler;
    textureDirty = true;
}

void Material::Apply(int frameIndex)
{
    // UBOプロパティの書き込み
    char* dstPtr = static_cast<char*>(materialUniformBuffersMapped[frameIndex]);
    for (const auto& propertyInfo : bufferInfo._propertyInfos)
    {
        auto floatValue = _floatMap.find(propertyInfo.name);
        if (floatValue != _floatMap.end())
        {
            memcpy(dstPtr + propertyInfo.offset, &floatValue->second, propertyInfo.size);
        }

        auto vec3Value = _vec3Map.find(propertyInfo.name);
        if (vec3Value != _vec3Map.end())
        {
            memcpy(dstPtr + propertyInfo.offset, &vec3Value->second, propertyInfo.size);
        }

        auto matrixValue = _matrixMap.find(propertyInfo.name);
        if (matrixValue != _matrixMap.end())
        {
            memcpy(dstPtr + propertyInfo.offset, &matrixValue->second, propertyInfo.size);
        }
    }

    // TODO: textureDirtyが立っている場合、SetTexture()で設定された
    // textureImageView / textureSampler を使ってテクスチャのDescriptorSetを更新する。
    // - shader->GetTextureInfoBySet(1) でテクスチャのbinding番号を取得する
    // - VkDescriptorImageInfo (imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) を組み立てる
    // - VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER のVkWriteDescriptorSetを
    //   materialDescriptorSets[0..count-1] それぞれに対して作り、device(メンバ変数)を使って
    //   vkUpdateDescriptorSetsで反映する (updateDescriptorSetsのテクスチャ部分が参考実装)
    // - 更新が終わったら textureDirty = false にする

    if (textureDirty)
    {
        std::vector<VkWriteDescriptorSet> descriptorWrites;

        // テクスチャ
        auto textureInfo = shader->GetTextureInfoBySet(1);
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet textureWrite{};
        textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        textureWrite.dstSet = materialDescriptorSets[frameIndex];
        textureWrite.dstBinding = textureInfo.binding;
        textureWrite.dstArrayElement = 0;
        textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureWrite.descriptorCount = 1;
        textureWrite.pImageInfo = &imageInfo;
        descriptorWrites.push_back(textureWrite);

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        textureDirty = false;
    }
}

const VkDescriptorSet* Material::DescriptorSet(int frameIndex) const
{
    return &materialDescriptorSets[frameIndex];
}

void Material::updateDescriptorSets(const VkDevice device)
{
    for (int i = 0; i < count; i++)
    {
        std::vector<VkWriteDescriptorSet> descriptorWrites;

        // バッファ
        VkDescriptorBufferInfo descBufferInfo{};
        descBufferInfo.buffer = materialUniformBuffers[i];
        descBufferInfo.offset = 0;
        descBufferInfo.range = bufferInfo.bufferSize;

        VkWriteDescriptorSet bufferWrite{};
        bufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bufferWrite.dstSet = materialDescriptorSets[i];
        bufferWrite.dstBinding = bufferInfo.binding;
        bufferWrite.dstArrayElement = 0;
        bufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bufferWrite.descriptorCount = 1;
        bufferWrite.pBufferInfo = &descBufferInfo;
        descriptorWrites.push_back(bufferWrite);

        // テクスチャ
        auto textureInfo = shader->GetTextureInfoBySet(1);
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet textureWrite{};
        textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        textureWrite.dstSet = materialDescriptorSets[i];
        textureWrite.dstBinding = textureInfo.binding;
        textureWrite.dstArrayElement = 0;
        textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureWrite.descriptorCount = 1;
        textureWrite.pImageInfo = &imageInfo;
        descriptorWrites.push_back(textureWrite);

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
