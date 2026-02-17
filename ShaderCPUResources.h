#pragma once
#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_reflect.hpp>

#include "VulkanBufferCreator.h"
#include "VulkanTextureCreator.h"

using ShaderStageFlags = uint8_t;

// どのシェーダステージで使われているかの8bitフラグ
enum ShaderStageFlagBits : uint8_t {
    ShaderStage_Unknown = 0,
    ShaderStage_Vertex = 1 << 0,
    ShaderStage_Fragment = 1 << 1,
    ShaderStage_Compute = 1 << 2,
    ShaderStage_Geometry = 1 << 3,
    ShaderStage_TessCtrl = 1 << 4,
    ShaderStage_TessEval = 1 << 5,
    // 必要に応じて追加
};

class ShaderBufferPropertyInfo {
public:
    std::string name;
    uint32_t offset;
    uint32_t size;
};

struct ShaderStageInputInfo
{
    uint32_t location;
    std::string name;
};

struct ShaderStageOutputInfo
{
    uint32_t location;
    std::string name;
};

struct ShaderBufferInfo {
    uint32_t set;
    uint32_t binding;
    uint32_t bufferSize;

    ShaderStageFlags stageFlags;

    std::vector<ShaderBufferPropertyInfo> _propertyInfos;

    ShaderBufferPropertyInfo GetPropertyInfoByName(const std::string& name) const {
        for (const auto& propertyInfo : _propertyInfos) {
            if (propertyInfo.name == name) {
                return propertyInfo;
            }
        }
        throw std::runtime_error("Property not found: " + name);
    }
};

struct ShaderTextureInfo {
    uint32_t set;
    uint32_t binding;
    std::string name;
    ShaderStageFlags stageFlags;
};

/// <summary>
/// Refrectionで取得したシェーダリソースの情報を格納する構造体
/// </summary>
struct ShaderResourceInfo {
    bool hasBufferResource;
    bool hasSamplerResource;


    std::vector<ShaderStageInputInfo> _stageInputInfos;
    std::vector<ShaderStageOutputInfo> _stageOutputInfos;
    std::vector<ShaderBufferInfo> _bufferInfos;
    std::vector<ShaderTextureInfo> _textureInfos;
};

/// <summary>
/// ShaderのCPU側のリソースを管理するクラス
/// BufferやTextureなどのリソースの作成と解放、DescriptorSetLayoutの作成、DescriptorSetへのリソースの紐づけなどを行う
/// </summary>
class ShaderCPUResource
{
    int count;

    ShaderResourceInfo vertexShaderResourceInfo;
    ShaderResourceInfo fragmentShaderResourceInfo;

    VkDescriptorSetLayout cameraDescriptorSetLayout;
    VkDescriptorSetLayout modelDescriptorSetLayout;

    std::vector<VkDescriptorSet> cameraDescriptorSets;
    std::vector<VkDescriptorSet> modelDescriptorSets;

    std::vector<VkBuffer> cameraUniformBuffers;
    std::vector<VkDeviceMemory> cameraUniformBufferMemories;
    std::vector<void*> cameraUniformBuffersMapped;

    std::vector<VkBuffer> modelUniformBuffers;
    std::vector<VkDeviceMemory> modelUniformBuffersMemories;
    std::vector<void*> modelUniformBuffersMapped;

public:
    ShaderCPUResource(const VkDevice device, const VulkanBufferCreator* bufferCreator, const VulkanTextureCreator* textureCreator, const VulkanResources* vulkanResources, VkDescriptorPool descriptorPool, int count)
    {
        // TODO BufferCreatorとVulkanResourcesは1つのクラスでいいのでは？

        this->count = count;
        vertexShaderResourceInfo = ShaderReflect("shaders/vert.spv");
        fragmentShaderResourceInfo = ShaderReflect("shaders/frag.spv");

        // Uniform Bufferの作成
        int cameraBufferIndex = 0; // TODO バッファの並び順は決まってないので、ShaderReflectで取得した情報をもとにどのバッファがカメラ用でどのバッファがモデル用かを判断する必要がある
        int modelBufferIndex = 1;
        auto cameraBufferInfo = vertexShaderResourceInfo._bufferInfos[cameraBufferIndex];
        auto modelBufferInfo = vertexShaderResourceInfo._bufferInfos[modelBufferIndex];
        VkDeviceSize cameraBufferSize = cameraBufferInfo.bufferSize;
        cameraUniformBuffers = bufferCreator->createUniformBuffers(cameraBufferSize, count, cameraUniformBufferMemories, cameraUniformBuffersMapped);

        VkDeviceSize modelBufferSize = modelBufferInfo.bufferSize;
        modelUniformBuffers = bufferCreator->createUniformBuffers(modelBufferSize, count, modelUniformBuffersMemories, modelUniformBuffersMapped);

        cameraDescriptorSetLayout = createCameraDescriptorSetLayout(device);
        modelDescriptorSetLayout = createModelDescriptorSetLayout(vertexShaderResourceInfo, fragmentShaderResourceInfo, device);
        cameraDescriptorSets = vulkanResources->createDescriptorSets(device, descriptorPool, cameraDescriptorSetLayout, count);
        modelDescriptorSets = vulkanResources->createDescriptorSets(device, descriptorPool, modelDescriptorSetLayout, count);

        // バッファとテクスチャとDescriptorSetsの紐づけ
        updateCameraDescriptorSets(device, cameraUniformBuffers, cameraBufferInfo, 0, cameraDescriptorSets);
        updateModelDescriptorSets(device, textureCreator, modelUniformBuffers, modelDescriptorSets, vertexShaderResourceInfo, fragmentShaderResourceInfo);
    }

    void Release(VkDevice device)
    {
        for (size_t i = 0; i < count; i++) {
            vkDestroyBuffer(device, cameraUniformBuffers[i], nullptr);
            vkFreeMemory(device, cameraUniformBufferMemories[i], nullptr);
            vkDestroyBuffer(device, modelUniformBuffers[i], nullptr);
            vkFreeMemory(device, modelUniformBuffersMemories[i], nullptr);
        }

        vkDestroyDescriptorSetLayout(device, modelDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, cameraDescriptorSetLayout, nullptr);
    }

    const ShaderResourceInfo VertexShaderResourceInfo() const {
        return vertexShaderResourceInfo;
    }

    const ShaderResourceInfo FragmentShaderResourceInfo() const {
        return fragmentShaderResourceInfo;
    }

    const VkDescriptorSetLayout CameraDescriptorSetLayout() const {
        return cameraDescriptorSetLayout;
    }

    const VkDescriptorSetLayout ModelDescriptorSetLayout() const {
        return modelDescriptorSetLayout;
    }

    const VkBuffer CameraUniformBuffers(int frameIndex) const {
        return cameraUniformBuffers[frameIndex];
    }
    void* CameraUniformBuffersMapped(int frameIndex) {
        return cameraUniformBuffersMapped[frameIndex];
    }
    const std::vector<VkBuffer> ModelUniformBuffers() const {
        return modelUniformBuffers;
    }
    void* ModelUniformBuffersMapped(int frameIndex) {
        return modelUniformBuffersMapped[frameIndex];
    }
    const VkDescriptorSet* CameraDescriptorSet(int frameIndex) const {
        return &cameraDescriptorSets[frameIndex];
    }
    const VkDescriptorSet* ModelDescriptorSet(int frameIndex) const {
        return &modelDescriptorSets[frameIndex];
    }

private:

    ShaderStageFlags GetShaderStageFlag(spv::ExecutionModel model)
    {
        // シェーダーステージに対応するフラグを返す
        switch (model)
        {
        case spv::ExecutionModelVertex:
            return ShaderStage_Vertex;
        case spv::ExecutionModelFragment:
            return ShaderStage_Fragment;
        case spv::ExecutionModelGLCompute:
            return ShaderStage_Compute;
        case spv::ExecutionModelGeometry:
            return ShaderStage_Geometry;
        case spv::ExecutionModelTessellationControl:
            return ShaderStage_TessCtrl;
        case spv::ExecutionModelTessellationEvaluation:
            return ShaderStage_TessEval;
        default:
            return ShaderStage_Unknown;
        }
    }

    ShaderResourceInfo ShaderReflect(std::string shaderPath)
    {
        // Read SPIR-V from disk or similar.
        std::vector<uint32_t> spirv_binary = readFileAsUint32(shaderPath);

        spirv_cross::CompilerGLSL glsl(std::move(spirv_binary));

        // The SPIR-V is now parsed, and we can perform reflection on it.
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();

        auto shaderStageType = GetShaderStageFlag(glsl.get_execution_model());

        ShaderResourceInfo shaderResourceInfo{};

        shaderResourceInfo._stageInputInfos.resize(resources.stage_inputs.size());
        std::cout << "-----Stage Input Infos" << std::endl;
        for (int i = 0; i < resources.stage_inputs.size(); ++i)
        {
            auto& stageInput = resources.stage_inputs[i];
            ShaderStageInputInfo inputInfo{};
            inputInfo.location = glsl.get_decoration(stageInput.id, spv::DecorationLocation);
            inputInfo.name = stageInput.name;
            shaderResourceInfo._stageInputInfos[i] = inputInfo;

            printf("\tInput %s at location = %u\n", stageInput.name.c_str(), inputInfo.location);
        }

        shaderResourceInfo._stageOutputInfos.resize(resources.stage_outputs.size());
        std::cout << "-----Stage Output Infos" << std::endl;
        for (int i = 0; i < resources.stage_outputs.size(); ++i)
        {
            auto& stageOutput = resources.stage_outputs[i];
            ShaderStageOutputInfo outputInfo{};
            outputInfo.location = glsl.get_decoration(stageOutput.id, spv::DecorationLocation);
            outputInfo.name = stageOutput.name;
            shaderResourceInfo._stageOutputInfos[outputInfo.location] = outputInfo;
            printf("\tOutput %s at location = %u\n", stageOutput.name.c_str(), outputInfo.location);
        }

        std::cout << "-----Uniform Buffer Infos" << std::endl;
        shaderResourceInfo.hasBufferResource = resources.uniform_buffers.size() > 0;
        if (shaderResourceInfo.hasBufferResource)
        {
            shaderResourceInfo._bufferInfos.resize(resources.uniform_buffers.size());
            for (int i = 0; i < resources.uniform_buffers.size(); ++i)
            {
                ShaderBufferInfo bufferInfo{};
                auto& uniformBuffer = resources.uniform_buffers[i];
                unsigned location = glsl.get_decoration(uniformBuffer.id, spv::DecorationLocation);
                std::string name = uniformBuffer.name;
                std::cout << "\tInput: " << name << ", Location: " << location << std::endl;
                auto propertyType = glsl.get_type(uniformBuffer.base_type_id);

                bufferInfo.stageFlags = shaderStageType;
                bufferInfo.set = glsl.get_decoration(uniformBuffer.id, spv::DecorationDescriptorSet);
                bufferInfo.binding = glsl.get_decoration(uniformBuffer.id, spv::DecorationBinding);
                bufferInfo.bufferSize = glsl.get_declared_struct_size(propertyType);

                auto propertySize = propertyType.member_types.size();
                bufferInfo._propertyInfos.resize(propertySize);
                for (int propertyIndex = 0; propertyIndex < propertySize; ++propertyIndex)
                {
                    ShaderBufferPropertyInfo propertyInfo{};
                    propertyInfo.name = glsl.get_member_name(uniformBuffer.base_type_id, propertyIndex);
                    propertyInfo.offset = glsl.type_struct_member_offset(propertyType, propertyIndex);
                    propertyInfo.size = glsl.get_declared_struct_member_size(propertyType, propertyIndex);
                    bufferInfo._propertyInfos[propertyIndex] = propertyInfo;
                }

                shaderResourceInfo._bufferInfos[i] = bufferInfo;
            }
        }

        std::cout << "-----Sampled Image Infos" << std::endl;
        shaderResourceInfo._textureInfos.resize(resources.sampled_images.size());
        shaderResourceInfo.hasSamplerResource = resources.sampled_images.size() > 0;
        for (int i = 0; i < resources.sampled_images.size(); ++i)
        {
            auto& image = resources.sampled_images[i];
            ShaderTextureInfo textureInfo{};
            textureInfo.stageFlags = shaderStageType;
            textureInfo.set = glsl.get_decoration(image.id, spv::DecorationDescriptorSet);
            textureInfo.binding = glsl.get_decoration(image.id, spv::DecorationBinding);
            textureInfo.name = image.name;
            shaderResourceInfo._textureInfos[i] = textureInfo;

            printf("Image %s at set = %u, binding = %u\n", image.name.c_str(), textureInfo.set, textureInfo.binding);
        }

        return shaderResourceInfo;
    }

    VkDescriptorSetLayout createCameraDescriptorSetLayout(const VkDevice device)
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        VkDescriptorSetLayout descriptorSetLayout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        return descriptorSetLayout;

    }

    VkDescriptorSetLayout createModelDescriptorSetLayout(const ShaderResourceInfo vertexShaderResourceInfo, const ShaderResourceInfo fragmentShaderResourceInfo, const VkDevice device)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings{};
        if (vertexShaderResourceInfo.hasBufferResource) {
            VkDescriptorSetLayoutBinding uboLayoutBinding{};
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            uboLayoutBinding.descriptorCount = 1; // 1でも問題ない...か？
            uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            uboLayoutBinding.pImmutableSamplers = nullptr; // Optional
            bindings.push_back(uboLayoutBinding);
        }

        // TODO フラグメントシェーダでしかテクスチャを使うことはほぼないのでこれで問題ないが、両方でテクスチャを使う場合もあるかも
        if (fragmentShaderResourceInfo.hasSamplerResource)
        {
            VkDescriptorSetLayoutBinding samplerLayoutBinding{};
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerLayoutBinding.pImmutableSamplers = nullptr;
            samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings.push_back(samplerLayoutBinding);
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();

        VkDescriptorSetLayout descriptorSetLayout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        return descriptorSetLayout;
    }

    void updateCameraDescriptorSets(const VkDevice device, const std::vector<VkBuffer> uniformBuffers, const ShaderBufferInfo shaderBufferInfo, const VkDeviceSize offset, const std::vector<VkDescriptorSet> uniformDescriptorSets)
    {
        size_t bufferCount = uniformBuffers.size();
        for (size_t i = 0; i < bufferCount; i++)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = offset;
            bufferInfo.range = shaderBufferInfo.bufferSize;

            VkWriteDescriptorSet descriptorWrites{};
            descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.dstSet = uniformDescriptorSets[i];
            descriptorWrites.dstBinding = 0;
            descriptorWrites.dstArrayElement = 0;
            descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.descriptorCount = 1;
            descriptorWrites.pBufferInfo = &bufferInfo;
            descriptorWrites.pImageInfo = nullptr; // Optional
            descriptorWrites.pTexelBufferView = nullptr; // Optional
            vkUpdateDescriptorSets(device, 1, &descriptorWrites, 0, nullptr);
        }
    }

    void updateModelDescriptorSets(const VkDevice device, const VulkanTextureCreator* textureCreator, const std::vector<VkBuffer> modelUniformBuffers, const std::vector<VkDescriptorSet> modelDescriptorSets, ShaderResourceInfo vertexShaderResourceInfo, ShaderResourceInfo fragmentShaderResourceInfo)
    {
        size_t bufferCount = modelUniformBuffers.size();
        for (size_t i = 0; i < bufferCount; i++)
        {
            std::vector<VkWriteDescriptorSet> descriptorWrites;

            // バッファ
            if (vertexShaderResourceInfo.hasBufferResource) {
                auto shaderBufferInfo = vertexShaderResourceInfo._bufferInfos[1]; // TODO 1番目がモデル用のバッファとは限らないので、ShaderReflectで取得した情報をもとにどのバッファがモデル用かを判断する必要がある
                VkWriteDescriptorSet writeDescriptorSet{};
                VkDescriptorBufferInfo bufferInfo{};
                bufferInfo.buffer = modelUniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = shaderBufferInfo.bufferSize;
                writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSet.dstSet = modelDescriptorSets[i];
                writeDescriptorSet.dstBinding = shaderBufferInfo.binding;
                writeDescriptorSet.dstArrayElement = 0;
                writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSet.descriptorCount = 1;
                writeDescriptorSet.pBufferInfo = &bufferInfo;
                descriptorWrites.push_back(writeDescriptorSet);
            }

            // テクスチャ
            if (fragmentShaderResourceInfo.hasSamplerResource) {
                VkWriteDescriptorSet writeDescriptorSet{};
                auto textureInfo = fragmentShaderResourceInfo._textureInfos[0];
                writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSet.dstSet = modelDescriptorSets[i];
                writeDescriptorSet.dstBinding = textureInfo.binding;
                writeDescriptorSet.dstArrayElement = 0;
                writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSet.descriptorCount = 1;
                VkDescriptorImageInfo imageInfo{};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = textureCreator->GetInvalidTextureImageView(); // 最初は無効なテクスチャ
                imageInfo.sampler = textureCreator->GetInvalidTextureSampler();
                writeDescriptorSet.pImageInfo = &imageInfo;
                descriptorWrites.push_back(writeDescriptorSet);
            }

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
};
