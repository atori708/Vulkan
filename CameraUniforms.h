#pragma once
#include <unordered_map>
#include <string>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Shader.h"
#include "VulkanBufferCreator.h"
#include "VulkanResources.h"

/// <summary>
/// カメラ(view/proj)用UBOとDescriptorSet(set=0)を管理するクラス
/// シーン全体で1つ持ち、Materialとは独立してApplyする
/// </summary>
class CameraUniforms
{
    int count;
    Shader* shader;
    ShaderBufferInfo bufferInfo;

    std::vector<VkDescriptorSet> cameraDescriptorSets;

    std::vector<VkBuffer> cameraUniformBuffers;
    std::vector<VkDeviceMemory> cameraUniformBufferMemories;
    std::vector<void*> cameraUniformBuffersMapped;

    std::unordered_map<std::string, glm::mat4x4> _matrixMap;

public:
    CameraUniforms(const VkDevice device, const VulkanBufferCreator* bufferCreator, const VulkanResources* vulkanResources, VkDescriptorPool descriptorPool, Shader* shader, int count);
    void Release(VkDevice device);
    void SetMatrix4x4(const std::string& name, const glm::mat4x4 matrix);
    void Apply(int frameIndex);
    const VkDescriptorSet* DescriptorSet(int frameIndex) const;

private:
    void updateDescriptorSets(const VkDevice device);
};
