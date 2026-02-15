#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spirv_cross/spirv_reflect.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <array>
#include <chrono>

#include "VulkanDevice.h"
#include "VulkanSwapChain.h"
#include "VulkanResources.h"
#include "VulkanCommandBuffer.h"
#include "VulkanBufferCreator.h"
#include "VulkanTextureCreator.h"

#include "IModelLoader.h"
#include "ModelLoaderByTinyObjLoader.h"
#include "ModelLoaderAssimp.h"

#include "Mesh.h"

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


struct UniformBufferObject {
    glm::mat4 model;
};

struct Camera {
    glm::vec3 position;
    glm::vec3 lookat;
    glm::vec3 up;
    float fov;
    float aspect;
    float nearClip;
    float farClip;
};

struct CameraUniformBufferObject {
    glm::mat4 view;
    glm::mat4 proj;
};

struct ShaderBufferPropertyInfo {
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
/// 値の更新を実際にシェーダに反映する処理
/// </summary>
/// <remarks>
/// 名前とかで何番目の値なのか覚えておいて、反映するタイミングでMap,Unmapをする
/// </remarks>
class ShaderPropertyApplier
{
private:
  ShaderResourceInfo vertexShaderResoureInfo;
  ShaderResourceInfo fragmentShaderResourceInfo;

  VkDescriptorSet cameraDescriptorSet;
  VkDescriptorSet modelDescriptorSet;

  std::vector<VkBuffer> cameraUniformBuffers;
  std::vector<VkDeviceMemory> cameraUniformBufferMemories;
  std::vector<void*> cameraUniformBuffersMapped;

  std::vector<VkBuffer> modelUniformBuffers;
  std::vector<VkDeviceMemory> modelUniformBuffersMemories;
  std::vector<void*> modelUniformBuffersMapped;

  std::unordered_map<std::string, glm::mat4x4> _matrixMap;

public:
    ShaderPropertyApplier(std::vector<VkBuffer> modelUniformBuffers,
        std::vector<VkDeviceMemory> modelUniformBuffersMemories,
        std::vector<void*> modelUniformBuffersMapped,
        ShaderResourceInfo vertexShaderResoureInfo,
        ShaderResourceInfo fragmentShaderResourceInfo)
		: modelUniformBuffers(modelUniformBuffers), modelUniformBuffersMemories(modelUniformBuffersMemories), modelUniformBuffersMapped(modelUniformBuffersMapped), vertexShaderResoureInfo(vertexShaderResoureInfo), fragmentShaderResourceInfo(fragmentShaderResourceInfo)
    {
		_matrixMap["view"] = glm::mat4x4(1.0f);
		_matrixMap["proj"] = glm::mat4x4(1.0f);
		_matrixMap["model"] = glm::mat4x4(1.0f);
	}

    void SetMatrix4x4(const std::string& name, const glm::mat4x4 matrix) {
		_matrixMap[name] = matrix;
    }

    void Apply(int frameIndex)
    {
		auto viewMatrix = _matrixMap["view"];
        auto projMatrix = _matrixMap["proj"];
        // モデル行列更新
		auto modelMatrix = _matrixMap["model"];
        char* dstPtr = static_cast<char*>(modelUniformBuffersMapped[frameIndex]);
		memcpy(dstPtr, &modelMatrix, sizeof(modelMatrix)); // 個々のサイズとかはShaderResourceInfoから取得して、適切なオフセットに書き込む必要がある
    }
};


class HelloTriangleApplication {
public :

    void run() {

        initWindow();
        initVulkan();
        auto swapChainExtent = vulkanSwapChain->GetSwapChainExtent();
        camera.position = glm::vec3(2.0f, 2.0f, 2.0f);
        camera.lookat = glm::vec3(0.0f, 0.0f, 0.0f);
        camera.up = glm::vec3(0.0f, 0.0f, 1.0f);
        camera.fov = 45.0f;
        camera.aspect = swapChainExtent.width / (float)swapChainExtent.height;
        camera.nearClip = 0.1f;
        camera.farClip = 10.0f;
        mainLoop();
        cleanup();
    }

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
            for(int i = 0; i < resources.uniform_buffers.size(); ++i)
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
                for(int propertyIndex = 0; propertyIndex < propertySize; ++propertyIndex)
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

private:
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;
    const int MAX_FRAMES_IN_FLIGHT = 2;
    const VkClearValue clearColor = { {{0.1f, 0.0f, 0.0f, 1.0f}} };

    GLFWwindow* window;
    bool frameBufferResized = false;

    Camera camera;

    VulkanDevice vulkanFacade;
    VulkanSwapChain* vulkanSwapChain;
    VulkanResources* vulkanResources;
    VulkanCommandBuffer* vulkanCommandBuffer;

    VulkanBufferCreator* bufferCreator;
    VulkanTextureCreator* textureCreator;

    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    float deviceMaxAnisotropy = 1;

    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;

    VkDescriptorSetLayout cameraDescriptorSetLayout;
    VkDescriptorSetLayout modelDescriptorSetLayout;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    VkCommandPool commandPool;
    VkCommandPool tempCommandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    VkBuffer vertexAndIndexBuffer;
    VkDeviceMemory vertexAndIndexBufferMemory;
    VkDeviceSize indexBufferOffset;

    std::vector<VkBuffer> cameraUniformBuffers;
    std::vector<VkDeviceMemory> cameraUniformBufferMemories;
    std::vector<void*> cameraUniformBuffersMapped;

    std::vector<VkBuffer> modelUniformBuffers;
    std::vector<VkDeviceMemory> modelUniformBuffersMemories;
    std::vector<void*> modelUniformBuffersMapped;

	ShaderPropertyApplier* shaderPropertyApplier;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> cameraDescriptorSets;
    std::vector<VkDescriptorSet> modelDescriptorSets;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    uint32_t currentFrame = 0;

    IModelLoader* modelLoader;

    Mesh* mesh;
    
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, resizedFrameBufferCallback);
    }

    static void resizedFrameBufferCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->frameBufferResized = true;
    }

    void initVulkan() {
        VulkanQueue queue = {};
        vulkanFacade.InitializeVulkan(window, queue);
        instance = vulkanFacade.GetInstance();
        physicalDevice = vulkanFacade.GetPhysicalDevice();
        logicalDevice = vulkanFacade.GetLogicalDevice();
        surface = vulkanFacade.GetSurface();
        graphicsQueue = vulkanFacade.GetGraphicsQueue();

        vulkanResources = new VulkanResources(logicalDevice);
        auto graphicsQueueFamilyIndicies = queue.FindGraphicsQueueFamilies(physicalDevice, surface);
        vulkanCommandBuffer = new VulkanCommandBuffer(instance, logicalDevice, graphicsQueue, MAX_FRAMES_IN_FLIGHT, graphicsQueueFamilyIndicies);
        vulkanSwapChain = new VulkanSwapChain(window, physicalDevice, logicalDevice, &queue, vulkanCommandBuffer, vulkanResources);
        bufferCreator = new VulkanBufferCreator(physicalDevice, logicalDevice, vulkanCommandBuffer);
        textureCreator = new VulkanTextureCreator(logicalDevice, physicalDevice, bufferCreator, vulkanResources, vulkanCommandBuffer, deviceMaxAnisotropy);

        vkGetDeviceQueue(logicalDevice, graphicsQueueFamilyIndicies.presentFamily.value(), 0, &presentQueue);

        VkSurfaceFormatKHR surfaceFormat{};
        surfaceFormat.format = VK_FORMAT_B8G8R8A8_SRGB;
        surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

        // 同期オブジェクトの作成
        createSyncObjects();

        // RenderPassの作成
        renderPass = createRenderPass(surfaceFormat.format);

        // SwapChainの作成(ImageViewとかデプスとかも)
        vulkanSwapChain->Initialize(surface, surfaceFormat, renderPass);

        auto vertexShaderResourceInfo = ShaderReflect("shaders/vert.spv");
		auto fragmentShaderResourceInfo = ShaderReflect("shaders/frag.spv");

        cameraDescriptorSetLayout = createCameraDescriptorSetLayout(logicalDevice);
        modelDescriptorSetLayout = createModelDescriptorSetLayout(vertexShaderResourceInfo, fragmentShaderResourceInfo, logicalDevice);

        // CommandPool、CommandBufferの作成
        commandPool = vulkanCommandBuffer->createCommandPool(graphicsQueueFamilyIndicies, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        tempCommandPool = vulkanCommandBuffer->createCommandPool(graphicsQueueFamilyIndicies, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
        commandBuffers = vulkanCommandBuffer->createCommandBuffers(commandPool);

        // シェーダモジュールとグラフィックスパイプラインの作成
        vertShaderModule = vulkanResources->createShaderModule("shaders/vert.spv");
        fragShaderModule = vulkanResources->createShaderModule("shaders/frag.spv");
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { cameraDescriptorSetLayout, modelDescriptorSetLayout };
        graphicsPipeline = createGraphicsPipeline(renderPass, descriptorSetLayouts, vertShaderModule, fragShaderModule);

        modelLoader = new ModelLoaderAssimp();
        // 頂点、インデックスバッファの作成
        mesh = modelLoader->loadModel("Assets/viking_room.obj");
        vertexAndIndexBuffer = bufferCreator->createVertexAndIndexBuffer(mesh->vertices, mesh->indices, indexBufferOffset, vertexAndIndexBufferMemory);

		// Uniform Bufferの作成
        int cameraBufferIndex = 0; // TODO バッファの並び順は決まってないので、ShaderReflectで取得した情報をもとにどのバッファがカメラ用でどのバッファがモデル用かを判断する必要がある
		int modelBufferIndex = 1;
		auto cameraBufferInfo = vertexShaderResourceInfo._bufferInfos[cameraBufferIndex];
		auto modelBufferInfo = vertexShaderResourceInfo._bufferInfos[modelBufferIndex];
        VkDeviceSize cameraBufferSize = cameraBufferInfo.bufferSize;
        cameraUniformBuffers = bufferCreator->createUniformBuffers(cameraBufferSize, MAX_FRAMES_IN_FLIGHT, cameraUniformBufferMemories, cameraUniformBuffersMapped);

        VkDeviceSize modelBufferSize = modelBufferInfo.bufferSize; 
        modelUniformBuffers = bufferCreator->createUniformBuffers(modelBufferSize, MAX_FRAMES_IN_FLIGHT, modelUniformBuffersMemories, modelUniformBuffersMapped);


        // テクスチャの作成
        textureImage = textureCreator->createTextureImage("Assets/viking_room.png", textureImageMemory);
        textureImageView = vulkanResources->createImageView2D(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
        textureSampler = textureCreator->createTextureSampler(deviceMaxAnisotropy);

		// DescriptorPoolとDescriptorSetの作成
		descriptorPool = createDescriptorPool(logicalDevice, MAX_FRAMES_IN_FLIGHT * 2); // DescriptorSetsはCamera用とModel用で2種類あるので、2倍の数を作成する必要がある
        cameraDescriptorSets = createDescriptorSets(logicalDevice, descriptorPool, cameraDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT);
        modelDescriptorSets = createDescriptorSets(logicalDevice, descriptorPool, modelDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT);

		shaderPropertyApplier = new ShaderPropertyApplier(modelUniformBuffers,
            modelUniformBuffersMemories, 
            modelUniformBuffersMapped, 
            vertexShaderResourceInfo, 
            fragmentShaderResourceInfo);

		// バッファとテクスチャとDescriptorSetsの紐づけ
        updateCameraDescriptorSets(cameraUniformBuffers, cameraBufferInfo, 0, cameraDescriptorSets);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;
        updateModelDescriptorSets(modelUniformBuffers, 0, modelBufferSize, imageInfo, modelDescriptorSets);
    }

    void updateCameraDescriptorSets(const std::vector<VkBuffer> uniformBuffers, const ShaderBufferInfo shaderBufferInfo, const VkDeviceSize offset, const std::vector<VkDescriptorSet> uniformDescriptorSets)
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
            vkUpdateDescriptorSets(logicalDevice, 1, &descriptorWrites, 0, nullptr);
        }
    }

    void updateModelDescriptorSets(const std::vector<VkBuffer> modelUniformBuffers, const VkDeviceSize offset, const VkDeviceSize range, const VkDescriptorImageInfo imageInfo, const std::vector<VkDescriptorSet> modelDescriptorSets)
    {
        size_t bufferCount = modelUniformBuffers.size();
        for (size_t i = 0; i < bufferCount; i++)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = modelUniformBuffers[i];
            bufferInfo.offset = offset;
            bufferInfo.range = range;

            // buffer更新
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = modelDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            // image更新
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = modelDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame(currentFrame);
            currentFrame++;
        }

        vkDeviceWaitIdle(logicalDevice);
    }

#pragma region 描画
    void drawFrame(uint32_t currentFrame)
    {
        uint32_t frameIndex = currentFrame % MAX_FRAMES_IN_FLIGHT;
        vkWaitForFences(logicalDevice, 1, &inFlightFences[frameIndex], VK_TRUE, UINT64_MAX);

        auto swapChain = vulkanSwapChain->GetSwapChain();
        auto swapChainExtent = vulkanSwapChain->GetSwapChainExtent();

        // Uniform Bufferの更新
        updateCameraUniformBuffers(cameraUniformBuffersMapped, 0, frameIndex, camera);
        updateModelUniformBuffers(modelUniformBuffersMapped, 0, frameIndex);

        // SwapChainが古くなっている場合は再作成する
        uint32_t imageIndex;
        auto result = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores[frameIndex], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            vulkanSwapChain->recreateSwapChain(surface, renderPass, swapChainExtent);
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(logicalDevice, 1, &inFlightFences[frameIndex]);

        vkResetCommandBuffer(commandBuffers[frameIndex], 0);
        recordCommandBuffer(frameIndex, commandBuffers[frameIndex], renderPass, imageIndex, swapChainExtent);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[frameIndex] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[frameIndex];

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[frameIndex] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[frameIndex]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (frameBufferResized || result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            vulkanSwapChain->recreateSwapChain(surface, renderPass, swapChainExtent);
            frameBufferResized = false;
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to present swap chain image!");
        }
    }

    void updateCameraUniformBuffers(const std::vector<void*> uniformBufferMapped, const size_t offset, const uint32_t frameIndex, const Camera& camera)
    {
        CameraUniformBufferObject cubo{};
        cubo.view = glm::lookAt(camera.position, camera.lookat, camera.up);
        cubo.proj = glm::perspective(glm::radians(camera.fov), camera.aspect, camera.nearClip, camera.farClip);
        cubo.proj[1][1] *= -1; // GLMはY座標が反転しているので、Vulkanに合わせて反転する
        char* dstPtr = static_cast<char*>(uniformBufferMapped[frameIndex]) + offset;
        memcpy(dstPtr, &cubo, sizeof(cubo));
    }

    void updateModelUniformBuffers(const std::vector<void*> uniformBufferMapped, const size_t offset, const uint32_t frameIndex)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};

        // 回転させる
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		shaderPropertyApplier->SetMatrix4x4("model", ubo.model); // 毎フレーム更新する値はShaderPropertyApplierにセットしておいて、ApplyのタイミングでまとめてUniform Bufferに反映させる
        /*char* dstPtr = static_cast<char*>(uniformBufferMapped[frameIndex]) + offset;
        memcpy(dstPtr, &ubo, sizeof(ubo));*/

		shaderPropertyApplier->Apply(frameIndex);
    }

    void recordCommandBuffer(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkRenderPass renderPass, uint32_t swapChainBufferIndex, VkExtent2D swapChainExtent)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0] = clearColor;
        clearValues[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = vulkanSwapChain->GetFrameBuffer(swapChainBufferIndex);
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &cameraDescriptorSets[frameIndex], 0, nullptr);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, &modelDescriptorSets[frameIndex], 0, nullptr);

        VkBuffer vertexBuffers[] = { vertexAndIndexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, vertexAndIndexBuffer, indexBufferOffset, VK_INDEX_TYPE_UINT16);

        //vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mesh->indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
#pragma endregion

    /// <summary>
    /// お片付け
    /// </summary>
    void cleanup()
    {
        delete mesh;
        delete bufferCreator;
        delete textureCreator;

        vulkanCommandBuffer->Release();
        delete vulkanCommandBuffer;

        vkDestroySampler(logicalDevice, textureSampler, nullptr);
        vkDestroyImageView(logicalDevice, textureImageView, nullptr);

        vkDestroyImage(logicalDevice, textureImage, nullptr);
        vkFreeMemory(logicalDevice, textureImageMemory, nullptr);

        vkDestroyDescriptorSetLayout(logicalDevice, modelDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(logicalDevice, cameraDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr); // DescriptorSetsはDescriptorPoolと一緒に破棄されるので個別に破棄する必要はない

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(logicalDevice, cameraUniformBuffers[i], nullptr);
            vkFreeMemory(logicalDevice, cameraUniformBufferMemories[i], nullptr);
            vkDestroyBuffer(logicalDevice, modelUniformBuffers[i], nullptr);
            vkFreeMemory(logicalDevice, modelUniformBuffersMemories[i], nullptr);
        }

        vkDestroyBuffer(logicalDevice, vertexAndIndexBuffer, nullptr);
        vkFreeMemory(logicalDevice, vertexAndIndexBufferMemory, nullptr);

        destroySyncObjects();

        vulkanSwapChain->Cleanup();
        delete vulkanSwapChain;

        vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
        vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);

        delete vulkanResources;
        vulkanFacade.CleanupVulkan();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    VkResult CreateDebugUtilsObjectNameEXT(VkDevice device, const VkDebugUtilsObjectNameInfoEXT* pNameInfo) {
        auto func = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
        if (func != nullptr) {
            return func(device, pNameInfo);
        }
        else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    QueueFamilyIndices findGraphicsQueueFamilies(VkPhysicalDevice physicalDevice) {
        QueueFamilyIndices indices;
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // グラフィクスキューを探す
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            // Presentキューを探す
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.IsComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    /// <summary>
    /// サーフェスを作成する
    /// </summary>
    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    /// <summary>
    /// RenderPass作成
    /// </summary>
    /// <param name="swapChainImageFormat"></param>
    /// <returns></returns>
    VkRenderPass createRenderPass(VkFormat swapChainImageFormat)
    {
        // Color Attachment
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // SubPass
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.colorAttachmentCount = 1;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        // Dependency
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPass renderPass;
        VkRenderPassCreateInfo renderPassInfo{};
        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        return renderPass;
    }

#pragma region  デプスステンシル


    VkFormat findDepthFormat()
    {
        return findSupportedFormat(
            physicalDevice,
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilFormat(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
#pragma endregion

#pragma region グラフィクスパイプラインの作成
    VkPipeline createGraphicsPipeline(const VkRenderPass renderPass, const std::vector<VkDescriptorSetLayout> descriptorSetLayouts, const VkShaderModule vertShaderModule, const VkShaderModule fragShaderModule)
    {
        // シェーダの準備
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Dynamic State
        // PSOの一部状態を動的に変更できる機能
        // 逆にこの項目は常に設定
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // Vertex Input
        auto bindingDescription = Vertex::getVkVertexInputBindingDescription();
        auto attributeDescriptions = Vertex::getVkVertexInputAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        // Input Assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE; // VK_TRUE�ɂ���ƁAtopology��STRIP�ɂ����Ƃ���LINE�ƃ|���S�������ȃC���f�b�N�X�l�Ńv���~�e�B�u�𕪊��ł���...�H�炵��

        // Viewport and Scissor
        auto swapChainExtent = vulkanSwapChain->GetSwapChainExtent();
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        // Rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo{};
        rasterizerCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizerCreateInfo.depthClampEnable = VK_FALSE; // VK_TRUE�ɂ���ƁA�N���b�v��Ԃ̊O�ɏo���[�x���N�����v����(�j���͂��Ȃ�)�B�V���h�E�}�b�v�Ȃǂ̃P�[�X�Ŗ��ɗ��炵��
        rasterizerCreateInfo.rasterizerDiscardEnable = VK_FALSE; // VK_TRUE�ɂ���ƁA�t���O�����g�V�F�[�_�܂ōs�����Ƀv���~�e�B�u��j������B���_���������������ꍇ�ȂǂɎg��
        rasterizerCreateInfo.polygonMode = VK_POLYGON_MODE_FILL; // �|���S���̕`�惂�[�h�BLINE��POINT������
        rasterizerCreateInfo.lineWidth = 1.0f;
        rasterizerCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizerCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizerCreateInfo.depthBiasEnable = VK_FALSE;
        rasterizerCreateInfo.depthBiasConstantFactor = 0.0f; // Optional
        rasterizerCreateInfo.depthBiasClamp = 0.0f; // Optional
        rasterizerCreateInfo.depthBiasSlopeFactor = 0.0f; // Optional

        // MSAA
        VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo{};
        multisamplingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisamplingCreateInfo.sampleShadingEnable = VK_FALSE;
        multisamplingCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisamplingCreateInfo.minSampleShading = 1.0f; // Optional
        multisamplingCreateInfo.pSampleMask = nullptr; // Optional
        multisamplingCreateInfo.alphaToCoverageEnable = VK_FALSE; // Optional
        multisamplingCreateInfo.alphaToOneEnable = VK_FALSE; // Optional

        // BlendState
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo colorBlendStateInfo{};
        colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendStateInfo.logicOpEnable = VK_FALSE;
        colorBlendStateInfo.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlendStateInfo.pAttachments = &colorBlendAttachment;
        colorBlendStateInfo.attachmentCount = 1;
        colorBlendStateInfo.blendConstants[0] = 0.0f; // Optional
        colorBlendStateInfo.blendConstants[1] = 0.0f; // Optional
        colorBlendStateInfo.blendConstants[2] = 0.0f; // Optional
        colorBlendStateInfo.blendConstants[3] = 0.0f; // Optional

        // Pipeline Layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo{};
        depthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilCreateInfo.depthTestEnable = VK_TRUE;
        depthStencilCreateInfo.depthWriteEnable = VK_TRUE;
        depthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE; // minDepthBoundsとmaxDepthを使う場合にVK_TRUEにする 範囲外はテストに失敗する
        depthStencilCreateInfo.stencilTestEnable = VK_FALSE; // ステンシルテストを使う場合にVK_TRUEにする

        // パイプラインを作成
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizerCreateInfo;
        pipelineInfo.pMultisampleState = &multisamplingCreateInfo;
        pipelineInfo.pDepthStencilState = &depthStencilCreateInfo;
        pipelineInfo.pColorBlendState = &colorBlendStateInfo;
        pipelineInfo.pDynamicState = &dynamicState; // Optional
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional

        VkPipeline graphicsPipeline;
        if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        return graphicsPipeline;
    }
#pragma endregion

#pragma region ディスクリプタ
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

    VkDescriptorPool createDescriptorPool(const VkDevice device, const size_t maxSets)
    {
        std::array< VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(maxSets);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(maxSets);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(maxSets);
        VkDescriptorPool descriptorPool;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

        return descriptorPool;
    }

    std::vector<VkDescriptorSet> createDescriptorSets(const VkDevice device, const VkDescriptorPool descriptorPool, const VkDescriptorSetLayout descriptorSetLayout, const uint32_t descriptorCount)
    {
        std::vector<VkDescriptorSetLayout> layouts(descriptorCount, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(descriptorCount);
        allocInfo.pSetLayouts = layouts.data();

        std::vector<VkDescriptorSet> modelDescriptorSets(descriptorCount);
        auto result = vkAllocateDescriptorSets(device, &allocInfo, modelDescriptorSets.data());

        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        return modelDescriptorSets;
    }
#pragma endregion

#pragma region 同期オブジェクト
    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void destroySyncObjects()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
        }
    }
#pragma endregion

    /// <summary>
    /// サポートしているフォーマットを<paramref name="candidates"/>の中から探す
    /// </summary>
    VkFormat findSupportedFormat(const VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}