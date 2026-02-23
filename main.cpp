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

#include "VulkanContext.h"
#include "VulkanSwapChain.h"
#include "VulkanResources.h"
#include "VulkanCommandBuffer.h"
#include "ShaderCPUResources.h"
#include "ShaderPropertyApplier.h"
#include "VulkanBufferCreator.h"
#include "VulkanTextureCreator.h"

#include "IModelLoader.h"
#include "ModelLoaderByTinyObjLoader.h"
#include "ModelLoaderAssimp.h"

#include "Mesh.h"

struct Camera {
    glm::vec3 position;
    glm::vec3 lookat;
    glm::vec3 up;
    float fov;
    float aspect;
    float nearClip;
    float farClip;
};

class HelloTriangleApplication
{
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

private:

    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;
    const int MAX_FRAMES_IN_FLIGHT = 2;
    const VkClearValue clearColor = { {{0.1f, 0.0f, 0.0f, 1.0f}} };

    GLFWwindow* window;
    bool frameBufferResized = false;

    Camera camera;

    VulkanContext vulkanFacade;
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

    ShaderCPUResource* shaderCPUResource;
	ShaderPropertyApplier* shaderPropertyApplier;

    VkDescriptorPool descriptorPool;

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

        // DescriptorPoolとDescriptorSetの作成
        descriptorPool = createDescriptorPool(logicalDevice, MAX_FRAMES_IN_FLIGHT * 2); // DescriptorSetsはCamera用とModel用で2種類あるので、2倍の数を作成する必要がある

        shaderCPUResource = new ShaderCPUResource(logicalDevice, bufferCreator, textureCreator, vulkanResources, descriptorPool, MAX_FRAMES_IN_FLIGHT);
        shaderPropertyApplier = new ShaderPropertyApplier(shaderCPUResource);

        // CommandPool、CommandBufferの作成
        commandPool = vulkanCommandBuffer->createCommandPool(graphicsQueueFamilyIndicies, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
        tempCommandPool = vulkanCommandBuffer->createCommandPool(graphicsQueueFamilyIndicies, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
        commandBuffers = vulkanCommandBuffer->createCommandBuffers(commandPool);

        // シェーダモジュールとグラフィックスパイプラインの作成
        vertShaderModule = vulkanResources->createShaderModule("shaders/vert.spv");
        fragShaderModule = vulkanResources->createShaderModule("shaders/frag.spv");
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { shaderCPUResource->CameraDescriptorSetLayout(), shaderCPUResource->ModelDescriptorSetLayout()};
        graphicsPipeline = createGraphicsPipeline(renderPass, descriptorSetLayouts, vertShaderModule, fragShaderModule);

        modelLoader = new ModelLoaderAssimp();
        // 頂点、インデックスバッファの作成
        mesh = modelLoader->loadModel("Assets/viking_room.obj");
        vertexAndIndexBuffer = bufferCreator->createVertexAndIndexBuffer(mesh->vertices, mesh->indices, indexBufferOffset, vertexAndIndexBufferMemory);

        // テクスチャの作成
        textureImage = textureCreator->createTextureImage("Assets/viking_room.png", textureImageMemory);
        textureImageView = vulkanResources->createImageView2D(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
        textureSampler = textureCreator->createTextureSampler(deviceMaxAnisotropy);
    }

    /// <summary>
    /// テクスチャの差し替え
    /// </summary>
    /// <param name="modelUniformBuffers"></param>
    /// <param name="imageInfo"></param>
    /// <param name="modelDescriptorSets"></param>
    void updateImageDescriptorSets(const int bufferCount, const ShaderCPUResource* shaderResource, const VkDescriptorImageInfo imageInfo)
    {
        for (size_t i = 0; i < bufferCount; i++)
        {
            std::array<VkWriteDescriptorSet, 1> descriptorWrites{};

            // buffer更新
            VkDescriptorBufferInfo bufferInfo{};

            // image更新
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = *shaderResource->ModelDescriptorSet(i);
            descriptorWrites[0].dstBinding = 1;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &imageInfo;

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
        updateCameraUniformBuffers(shaderPropertyApplier, camera);
        updateModelUniformBuffers(shaderPropertyApplier);
        shaderPropertyApplier->Apply(frameIndex);

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

		// しばらくたったらテクスチャを差し替える
        if (currentFrame == 2000)
        {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;
            updateImageDescriptorSets(MAX_FRAMES_IN_FLIGHT, shaderCPUResource, imageInfo);

            shaderPropertyApplier->SetVector3("color", glm::vec3(0,1,1));

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

    void updateCameraUniformBuffers(ShaderPropertyApplier* shaderPropertyApplier, const Camera& camera)
    {
        auto view = glm::lookAt(camera.position, camera.lookat, camera.up);
        auto proj = glm::perspective(glm::radians(camera.fov), camera.aspect, camera.nearClip, camera.farClip);
        proj[1][1] *= -1; // GLMはY座標が反転しているので、Vulkanに合わせて反転する
		shaderPropertyApplier->SetMatrix4x4("view", view);
		shaderPropertyApplier->SetMatrix4x4("proj", proj);
    }

    void updateModelUniformBuffers(ShaderPropertyApplier* shaderPropertyApplier)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        // 回転させる
        auto model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		shaderPropertyApplier->SetMatrix4x4("model", model);
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

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, shaderCPUResource->CameraDescriptorSet(frameIndex), 0, nullptr);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 1, 1, shaderCPUResource->ModelDescriptorSet(frameIndex), 0, nullptr);

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

      
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr); // DescriptorSetsはDescriptorPoolと一緒に破棄されるので個別に破棄する必要はない

        vkDestroyBuffer(logicalDevice, vertexAndIndexBuffer, nullptr);
        vkFreeMemory(logicalDevice, vertexAndIndexBufferMemory, nullptr);

        shaderCPUResource->Release(logicalDevice);
        delete shaderCPUResource;
        delete shaderPropertyApplier;

        destroySyncObjects();

        vulkanSwapChain->Cleanup();
        delete vulkanSwapChain;

        vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
        vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);

        delete vulkanResources;
		vulkanResources = nullptr;
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