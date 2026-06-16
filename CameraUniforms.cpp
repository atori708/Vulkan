#include "CameraUniforms.h"

CameraUniforms::CameraUniforms(const VkDevice device, const VulkanBufferCreator* bufferCreator, const VulkanResources* vulkanResources, VkDescriptorPool descriptorPool, Shader* shader, int count)
    : count(count), shader(shader), bufferInfo(shader->GetBufferInfoBySet(0))
{
    cameraUniformBuffers = bufferCreator->createUniformBuffers(bufferInfo.bufferSize, count, cameraUniformBufferMemories, cameraUniformBuffersMapped);
    cameraDescriptorSets = vulkanResources->createDescriptorSets(device, descriptorPool, shader->CameraDescriptorSetLayout(), count);

    updateDescriptorSets(device);

    _matrixMap["view"] = glm::mat4x4(1.0f);
    _matrixMap["proj"] = glm::mat4x4(1.0f);
}

void CameraUniforms::Release(VkDevice device)
{
    for (int i = 0; i < count; i++) {
        vkDestroyBuffer(device, cameraUniformBuffers[i], nullptr);
        vkFreeMemory(device, cameraUniformBufferMemories[i], nullptr);
    }
}

void CameraUniforms::SetMatrix4x4(const std::string& name, const glm::mat4x4 matrix)
{
    _matrixMap[name] = matrix;
}

void CameraUniforms::Apply(int frameIndex)
{
    char* dstPtr = static_cast<char*>(cameraUniformBuffersMapped[frameIndex]);
    for (const auto& propertyInfo : bufferInfo._propertyInfos)
    {
        auto matrix = _matrixMap.find(propertyInfo.name);
        if (matrix != _matrixMap.end())
        {
            memcpy(dstPtr + propertyInfo.offset, &matrix->second, propertyInfo.size);
        }
    }
}

const VkDescriptorSet* CameraUniforms::DescriptorSet(int frameIndex) const
{
    return &cameraDescriptorSets[frameIndex];
}

void CameraUniforms::updateDescriptorSets(const VkDevice device)
{
    for (int i = 0; i < count; i++)
    {
        VkDescriptorBufferInfo descBufferInfo{};
        descBufferInfo.buffer = cameraUniformBuffers[i];
        descBufferInfo.offset = 0;
        descBufferInfo.range = bufferInfo.bufferSize;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = cameraDescriptorSets[i];
        descriptorWrite.dstBinding = bufferInfo.binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &descBufferInfo;
        descriptorWrite.pImageInfo = nullptr;
        descriptorWrite.pTexelBufferView = nullptr;
        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }
}
