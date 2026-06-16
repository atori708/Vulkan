#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <vulkan/vulkan.h>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_reflect.hpp>

#include "VulkanResourceUtility.h"

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

    ShaderBufferPropertyInfo GetPropertyInfoByName(const std::string& name) const;
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
/// シェーダのリフレクション情報とDescriptorSetLayoutを保持するクラス
/// シェーダごとに1つ生成され、CameraUniformsやMaterialから共有参照される
/// </summary>
class Shader
{
    ShaderResourceInfo vertexShaderResourceInfo;
    ShaderResourceInfo fragmentShaderResourceInfo;

    VkDescriptorSetLayout cameraDescriptorSetLayout;
    VkDescriptorSetLayout materialDescriptorSetLayout;

public:
    Shader(const VkDevice device);
    void Release(VkDevice device);

    const ShaderResourceInfo VertexShaderResourceInfo() const;
    const ShaderResourceInfo FragmentShaderResourceInfo() const;
    const VkDescriptorSetLayout CameraDescriptorSetLayout() const;
    const VkDescriptorSetLayout MaterialDescriptorSetLayout() const;

    /// <summary>
    /// vertex/fragmentシェーダのUniform Bufferの中から、指定したdescriptor set番号に
    /// 紐づくShaderBufferInfoを探して返す。
    /// </summary>
    /// <remarks>
    /// 旧実装では `_bufferInfos[0]` がカメラ用、`_bufferInfos[1]` がモデル用と決め打ちしていたが、
    /// set番号(shader.vertの`layout(set = ..., binding = ...)`)を基準に検索することで
    /// バッファの並び順に依存しないようにする。
    /// vertexShaderResourceInfo._bufferInfos と fragmentShaderResourceInfo._bufferInfos の
    /// 両方を走査し、ShaderBufferInfo::set が一致する要素を返す。見つからない場合はthrowする。
    /// </remarks>
    ShaderBufferInfo GetBufferInfoBySet(uint32_t set) const;

    /// <summary>
    /// vertex/fragmentシェーダのテクスチャ(sampler2D)の中から、指定したdescriptor set番号に
    /// 紐づくShaderTextureInfoを探して返す。
    /// </summary>
    ShaderTextureInfo GetTextureInfoBySet(uint32_t set) const;

private:
    ShaderStageFlags GetShaderStageFlag(spv::ExecutionModel model);
    ShaderResourceInfo ShaderReflect(std::string shaderPath);
    VkDescriptorSetLayout createCameraDescriptorSetLayout(const VkDevice device);
    VkDescriptorSetLayout createMaterialDescriptorSetLayout(const ShaderResourceInfo vertexShaderResourceInfo, const ShaderResourceInfo fragmentShaderResourceInfo, const VkDevice device);
};
