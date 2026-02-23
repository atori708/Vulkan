#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "VulkanContext.h"
#include "VulkanSwapChain.h"
#include "VulkanResources.h"
#include "VulkanCommandBuffer.h"
#include "VulkanBufferCreator.h"
#include "VulkanTextureCreator.h"
#include "ShaderCPUResources.h"
#include "ShaderPropertyApplier.h"

#include "IModelLoader.h"
#include "ModelLoaderByTinyObjLoader.h"
#include "ModelLoaderAssimp.h"
#include <VulkanStatus.h>

#include "Scene/Camera.h"

class VulkanApp
{
public:
    VulkanApp(GLFWwindow* window);
    ~VulkanApp();
	void Draw(int currentFrame);

	void OnWindowResized() { frameBufferResized = true; }

    const int MAX_FRAMES_IN_FLIGHT = 2;
    const VkClearValue clearColor = { {{0.1f, 0.0f, 0.0f, 1.0f}} };

private:
    bool frameBufferResized = false;

    VulkanContext vulkanContext;
	VulkanStatus*  vulkanStatus;
    VulkanSwapChain* vulkanSwapChain;
    VulkanResources* vulkanResources;
    VulkanCommandBuffer* vulkanCommandBuffer;

    VulkanBufferCreator* bufferCreator;
    VulkanTextureCreator* textureCreator;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkDescriptorPool descriptorPool;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkCommandPool commandPool;
    VkCommandPool tempCommandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
    VkBuffer vertexAndIndexBuffer;
    VkDeviceMemory vertexAndIndexBufferMemory;
    VkDeviceSize indexBufferOffset;


    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    ShaderCPUResource* shaderCPUResource;
    ShaderPropertyApplier* shaderPropertyApplier;

    IModelLoader* modelLoader;
    Mesh* mesh;

    Camera camera;


    void createSurface(GLFWwindow* window);

    VkRenderPass createRenderPass(VkFormat swapChainImageFormat);
    VkPipeline createGraphicsPipeline(const VkRenderPass renderPass, const std::vector<VkDescriptorSetLayout> descriptorSetLayouts, const VkShaderModule vertShaderModule, const VkShaderModule fragShaderModule);
    VkPipeline createGraphicsPipeline(const std::vector<VkDescriptorSetLayout> descriptorSetLayouts, const VkShaderModule vertShaderModule, const VkShaderModule fragShaderModule);

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
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void destroySyncObjects()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
    }

    void recordCommandBuffer(uint32_t frameIndex, VkCommandBuffer commandBuffer, uint32_t swapChainBufferIndex, VkExtent2D swapChainExtent);
    void recordCommandBuffer(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkRenderPass renderPass, uint32_t swapChainBufferIndex, VkExtent2D swapChainExtent);

    void updateCameraUniformBuffers(ShaderPropertyApplier* shaderPropertyApplier, const Camera& camera);
    void updateModelUniformBuffers(ShaderPropertyApplier* shaderPropertyApplier);
    void updateImageDescriptorSets(const int bufferCount, const ShaderCPUResource* shaderResource, const VkDescriptorImageInfo imageInfo);
};

