#pragma once
#include <unordered_map>
#include <string>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Shader.h"
#include "VulkanBufferCreator.h"
#include "VulkanTextureCreator.h"
#include "VulkanResources.h"

/// <summary>
/// モデル固有のプロパティ(色・行列・テクスチャ等)とDescriptorSet(set=1)を管理するクラス
/// UnityのMaterialに相当し、SetFloat/SetVector3/SetMatrix4x4/SetTextureで値を設定し、
/// Apply()でUBOおよびテクスチャのDescriptorSetへ反映する
/// </summary>
class Material
{
    VkDevice device;
    int count;
    Shader* shader;
    ShaderBufferInfo bufferInfo;

    std::vector<VkDescriptorSet> materialDescriptorSets;

    std::vector<VkBuffer> materialUniformBuffers;
    std::vector<VkDeviceMemory> materialUniformBufferMemories;
    std::vector<void*> materialUniformBuffersMapped;

    std::unordered_map<std::string, float> _floatMap;
    std::unordered_map<std::string, glm::vec3> _vec3Map;
    std::unordered_map<std::string, glm::mat4x4> _matrixMap;

    VkImageView textureImageView;
    VkSampler textureSampler;
    bool textureDirty = false;

public:
    Material(const VkDevice device, const VulkanBufferCreator* bufferCreator, const VulkanTextureCreator* textureCreator, const VulkanResources* vulkanResources, VkDescriptorPool descriptorPool, Shader* shader, int count);
    void Release(VkDevice device);
    void SetFloat(const std::string& name, const float value);
    void SetVector3(const std::string& name, const glm::vec3 vec3);
    void SetMatrix4x4(const std::string& name, const glm::mat4x4 matrix);

    /// <summary>
    /// 描画に使うテクスチャを差し替える。実際のDescriptorSetへの反映はApply()で行う
    /// </summary>
    void SetTexture(VkImageView imageView, VkSampler sampler);

    void Apply(int frameIndex);
    const VkDescriptorSet* DescriptorSet(int frameIndex) const;

private:
    void updateDescriptorSets(const VkDevice device);
};
