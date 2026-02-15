#pragma once
#include <map>
#include <vulkan/vulkan.h>
#include "ShaderCPUResources.h"

/// <summary>
/// 値の更新を実際にシェーダに反映する処理
/// </summary>
/// <remarks>
/// 名前とかで何番目の値なのか覚えておいて、反映するタイミングでMap,Unmapをする
/// </remarks>
class ShaderPropertyApplier
{
private:
    ShaderCPUResource* shaderCPUResource;
    ShaderResourceInfo vertexShaderResoureInfo;
    ShaderResourceInfo fragmentShaderResourceInfo;

    std::unordered_map<std::string, float> _floatMap;
    std::unordered_map<std::string, glm::vec3> _vec3Map;
    std::unordered_map<std::string, glm::mat4x4> _matrixMap;

public:
    ShaderPropertyApplier(ShaderCPUResource* shaderCPUResource)
        : shaderCPUResource(shaderCPUResource)
    {
        this->vertexShaderResoureInfo = shaderCPUResource->VertexShaderResourceInfo();
        this->fragmentShaderResourceInfo = shaderCPUResource->FragmentShaderResourceInfo();
        _matrixMap["view"] = glm::mat4x4(1.0f);
        _matrixMap["proj"] = glm::mat4x4(1.0f);
        _matrixMap["model"] = glm::mat4x4(1.0f);
    }

    void SetVector3(const std::string& name, const glm::vec3 vec3)
    {
        _vec3Map[name] = vec3;
    }

    void SetMatrix4x4(const std::string& name, const glm::mat4x4 matrix)
    {
        _matrixMap[name] = matrix;
    }

    void Apply(int frameIndex)
    {
        // カメラ行列更新
        char* dstPtr = static_cast<char*>(shaderCPUResource->CameraUniformBuffersMapped(frameIndex));
        auto bufferInfo = vertexShaderResoureInfo._bufferInfos[0]; // TODO 0番目がカメラ用のバッファとは限らない
        ApplyBuffer(dstPtr, bufferInfo, frameIndex);

        // モデル行列更新
        dstPtr = static_cast<char*>(shaderCPUResource->ModelUniformBuffersMapped(frameIndex));
        auto modelBufferInfo = vertexShaderResoureInfo._bufferInfos[1];
        ApplyBuffer(dstPtr, modelBufferInfo, frameIndex);

        // TODO 差分だけApplyするとかできたら最適化になりそう
    }

private:
    void ApplyBuffer(char* ptr, const ShaderBufferInfo bufferInfo, int frameIndex)
    {
        for (auto propertyInfo : bufferInfo._propertyInfos)
        {
            auto floatValue = _floatMap.find(propertyInfo.name);
            if (floatValue != _floatMap.end())
            {
                memcpy(ptr + propertyInfo.offset, &_floatMap[propertyInfo.name], propertyInfo.size);
            }

            auto matrix = _matrixMap.find(propertyInfo.name);
            if (matrix != _matrixMap.end())
            {
                memcpy(ptr + propertyInfo.offset, &_matrixMap[propertyInfo.name], propertyInfo.size);
            }

            auto vec3 = _vec3Map.find(propertyInfo.name);
            if (vec3 != _vec3Map.end())
            {
                memcpy(ptr + propertyInfo.offset, &_vec3Map[propertyInfo.name], propertyInfo.size);
            }
        }
    }
};