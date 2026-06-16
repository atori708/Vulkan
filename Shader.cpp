#include "Shader.h"

ShaderBufferPropertyInfo ShaderBufferInfo::GetPropertyInfoByName(const std::string& name) const
{
    for (const auto& propertyInfo : _propertyInfos) {
        if (propertyInfo.name == name) {
            return propertyInfo;
        }
    }
    throw std::runtime_error("Property not found: " + name);
}

Shader::Shader(const VkDevice device)
{
    vertexShaderResourceInfo = ShaderReflect("shaders/vert.spv");
    fragmentShaderResourceInfo = ShaderReflect("shaders/frag.spv");

    cameraDescriptorSetLayout = createCameraDescriptorSetLayout(device);
    materialDescriptorSetLayout = createMaterialDescriptorSetLayout(vertexShaderResourceInfo, fragmentShaderResourceInfo, device);
}

void Shader::Release(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, materialDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, cameraDescriptorSetLayout, nullptr);
}

const ShaderResourceInfo Shader::VertexShaderResourceInfo() const {
    return vertexShaderResourceInfo;
}

const ShaderResourceInfo Shader::FragmentShaderResourceInfo() const {
    return fragmentShaderResourceInfo;
}

const VkDescriptorSetLayout Shader::CameraDescriptorSetLayout() const {
    return cameraDescriptorSetLayout;
}

const VkDescriptorSetLayout Shader::MaterialDescriptorSetLayout() const {
    return materialDescriptorSetLayout;
}

ShaderBufferInfo Shader::GetBufferInfoBySet(uint32_t set) const
{
    for (const auto& bufferInfo : vertexShaderResourceInfo._bufferInfos)
    {
        if (bufferInfo.set == set) {
            return bufferInfo;
        }
    }

    throw std::runtime_error("Buffer Info not found for set:" + std::to_string(set));
}

ShaderTextureInfo Shader::GetTextureInfoBySet(uint32_t set) const
{
    for (const auto& textureInfo : vertexShaderResourceInfo._textureInfos) {
        if (textureInfo.set == set) {
            return textureInfo;
        }
    }
    for (const auto& textureInfo : fragmentShaderResourceInfo._textureInfos) {
        if (textureInfo.set == set) {
            return textureInfo;
        }
    }
    throw std::runtime_error("Texture info not found for set: " + std::to_string(set));
}

ShaderStageFlags Shader::GetShaderStageFlag(spv::ExecutionModel model)
{
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

ShaderResourceInfo Shader::ShaderReflect(std::string shaderPath)
{
    std::vector<uint32_t> spirv_binary = readFileAsUint32(shaderPath);
    spirv_cross::CompilerGLSL glsl(std::move(spirv_binary));
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

VkDescriptorSetLayout Shader::createCameraDescriptorSetLayout(const VkDevice device)
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

VkDescriptorSetLayout Shader::createMaterialDescriptorSetLayout(const ShaderResourceInfo vertexShaderResourceInfo, const ShaderResourceInfo fragmentShaderResourceInfo, const VkDevice device)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings{};
    if (vertexShaderResourceInfo.hasBufferResource) {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;
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
