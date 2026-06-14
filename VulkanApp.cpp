#include "VulkanApp.h"

VulkanApp::VulkanApp(GLFWwindow* window)
{
    VulkanQueue queue = {};
    vulkanContext.InitializeVulkan(window, queue);
    instance = vulkanContext.GetInstance();
    physicalDevice = vulkanContext.GetPhysicalDevice();
    device = vulkanContext.GetDevice();
    surface = vulkanContext.GetSurface();
    graphicsQueue = vulkanContext.GetGraphicsQueue();

	vulkanStatus = new VulkanStatus(physicalDevice);

    vulkanResources = new VulkanResources(device);
    auto graphicsQueueFamilyIndicies = queue.FindGraphicsQueueFamilies(physicalDevice, surface);
    vulkanCommandBuffer = new VulkanCommandBuffer(instance, device, graphicsQueue, MAX_FRAMES_IN_FLIGHT, graphicsQueueFamilyIndicies);
    vulkanSwapChain = new VulkanSwapChain(window, physicalDevice, device, &queue, vulkanCommandBuffer, vulkanResources);
    bufferCreator = new VulkanBufferCreator(physicalDevice, device, vulkanCommandBuffer);
    textureCreator = new VulkanTextureCreator(device, physicalDevice, bufferCreator, vulkanResources, vulkanCommandBuffer, vulkanStatus->GetMaxAnisotropy());

    vkGetDeviceQueue(device, graphicsQueueFamilyIndicies.presentFamily.value(), 0, &presentQueue);

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
    descriptorPool = vulkanContext.createDescriptorPool(device, MAX_FRAMES_IN_FLIGHT * 2); // DescriptorSetsはCamera用とModel用で2種類あるので、2倍の数を作成する必要がある

    shaderCPUResource = new ShaderCPUResource(device, bufferCreator, textureCreator, vulkanResources, descriptorPool, MAX_FRAMES_IN_FLIGHT);
    shaderPropertyApplier = new ShaderPropertyApplier(shaderCPUResource);

    // CommandPool、CommandBufferの作成
    commandPool = vulkanCommandBuffer->createCommandPool(graphicsQueueFamilyIndicies, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    tempCommandPool = vulkanCommandBuffer->createCommandPool(graphicsQueueFamilyIndicies, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    commandBuffers = vulkanCommandBuffer->createCommandBuffers(commandPool);

    // シェーダモジュールとグラフィックスパイプラインの作成
    vertShaderModule = vulkanResources->createShaderModule("shaders/vert.spv");
    fragShaderModule = vulkanResources->createShaderModule("shaders/frag.spv");
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { shaderCPUResource->CameraDescriptorSetLayout(), shaderCPUResource->ModelDescriptorSetLayout() };
    graphicsPipeline = createGraphicsPipeline(renderPass, descriptorSetLayouts, vertShaderModule, fragShaderModule);

    modelLoader = new ModelLoaderAssimp();
    // 頂点、インデックスバッファの作成
    mesh = modelLoader->loadModel("Assets/viking_room.obj");
    vertexAndIndexBuffer = bufferCreator->createVertexAndIndexBuffer(mesh->vertices, mesh->indices, indexBufferOffset, vertexAndIndexBufferMemory);

    // テクスチャの作成
    textureImage = textureCreator->createTextureImage("Assets/viking_room.png", textureImageMemory);
    textureImageView = vulkanResources->createImageView2D(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    textureSampler = textureCreator->createTextureSampler(vulkanStatus->GetMaxAnisotropy());

    auto swapChainExtent = vulkanSwapChain->GetSwapChainExtent();
    camera.position = glm::vec3(2.0f, 2.0f, 2.0f);
    camera.lookat = glm::vec3(0.0f, 0.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 0.0f, 1.0f);
    camera.fov = 45.0f;
    camera.aspect = swapChainExtent.width / (float)swapChainExtent.height;
    camera.nearClip = 0.1f;
    camera.farClip = 10.0f;
}

VulkanApp::~VulkanApp()
{
    vkDeviceWaitIdle(device);

    delete mesh;
    delete bufferCreator;
    delete textureCreator;

    vulkanCommandBuffer->Release();
    delete vulkanCommandBuffer;

    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);

    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr); // DescriptorSetsはDescriptorPoolと一緒に破棄されるので個別に破棄する必要はない

    vkDestroyBuffer(device, vertexAndIndexBuffer, nullptr);
    vkFreeMemory(device, vertexAndIndexBufferMemory, nullptr);

    shaderCPUResource->Release(device);
    delete shaderCPUResource;
    delete shaderPropertyApplier;

    destroySyncObjects();

    vulkanSwapChain->Cleanup();
    delete vulkanSwapChain;

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    delete vulkanResources;
    vulkanResources = nullptr;

    delete vulkanStatus;
	vulkanStatus = nullptr;

    vulkanContext.CleanupVulkan();
}

void VulkanApp::Draw(int currentFrame)
{
    uint32_t frameIndex = currentFrame % MAX_FRAMES_IN_FLIGHT;
    vkWaitForFences(device, 1, &inFlightFences[frameIndex], VK_TRUE, UINT64_MAX);

    auto swapChain = vulkanSwapChain->GetSwapChain();
    auto swapChainExtent = vulkanSwapChain->GetSwapChainExtent();

    // Uniform Bufferの更新
    updateCameraUniformBuffers(shaderPropertyApplier, camera);
    updateModelUniformBuffers(shaderPropertyApplier);
    shaderPropertyApplier->Apply(frameIndex);

    // SwapChainが古くなっている場合は再作成する
    uint32_t imageIndex;
    auto result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[frameIndex], VK_NULL_HANDLE, &imageIndex);
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
        // TODO なんかValdationLayerでエラー出てるっぽいので見直す
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;
        updateImageDescriptorSets(MAX_FRAMES_IN_FLIGHT, shaderCPUResource, imageInfo);

        shaderPropertyApplier->SetVector3("color", glm::vec3(0, 1, 1));
    }

    vkResetFences(device, 1, &inFlightFences[frameIndex]);

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

void VulkanApp::updateCameraUniformBuffers(ShaderPropertyApplier* shaderPropertyApplier, const Camera& camera)
{
    auto view = glm::lookAt(camera.position, camera.lookat, camera.up);
    auto proj = glm::perspective(glm::radians(camera.fov), camera.aspect, camera.nearClip, camera.farClip);
    proj[1][1] *= -1; // GLMはY座標が反転しているので、Vulkanに合わせて反転する
    shaderPropertyApplier->SetMatrix4x4("view", view);
    shaderPropertyApplier->SetMatrix4x4("proj", proj);
}

void VulkanApp::updateModelUniformBuffers(ShaderPropertyApplier* shaderPropertyApplier)
{
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    // 回転させる
    auto model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    shaderPropertyApplier->SetMatrix4x4("model", model);
}

/// <summary>
/// テクスチャの差し替え
/// </summary>
/// <param name="modelUniformBuffers"></param>
/// <param name="imageInfo"></param>
/// <param name="modelDescriptorSets"></param>
void VulkanApp::updateImageDescriptorSets(const int bufferCount, const ShaderCPUResource* shaderResource, const VkDescriptorImageInfo imageInfo)
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

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

/// <summary>
  /// サーフェスを作成する
  /// </summary>
void VulkanApp::createSurface(GLFWwindow* window)
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
VkRenderPass VulkanApp::createRenderPass(VkFormat swapChainImageFormat)
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
    depthAttachment.format = vulkanStatus->FindDepthFormat();
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

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    return renderPass;
}

#pragma region グラフィクスパイプラインの作成
VkPipeline VulkanApp::createGraphicsPipeline(const VkRenderPass renderPass, const std::vector<VkDescriptorSetLayout> descriptorSetLayouts, const VkShaderModule vertShaderModule, const VkShaderModule fragShaderModule)
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
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
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

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
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
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    return graphicsPipeline;
}

VkPipeline VulkanApp::createGraphicsPipeline(const std::vector<VkDescriptorSetLayout> descriptorSetLayouts, const VkShaderModule vertShaderModule, const VkShaderModule fragShaderModule)
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
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
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

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo{};
    depthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilCreateInfo.depthTestEnable = VK_TRUE;
    depthStencilCreateInfo.depthWriteEnable = VK_TRUE;
    depthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE; // minDepthBoundsとmaxDepthを使う場合にVK_TRUEにする 範囲外はテストに失敗する
    depthStencilCreateInfo.stencilTestEnable = VK_FALSE; // ステンシルテストを使う場合にVK_TRUEにする

    VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = vulkanSwapChain->GetColorFormat(),
        .depthAttachmentFormat = vulkanSwapChain->GetDepthFormat(),
    };

    // パイプラインを作成
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &pipelineRenderingCreateInfo;
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
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    VkPipeline graphicsPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    return graphicsPipeline;
}
#pragma endregion

void VulkanApp::recordCommandBuffer(uint32_t frameIndex, VkCommandBuffer commandBuffer, uint32_t swapChainBufferIndex, VkExtent2D swapChainExtent)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    vulkanCommandBuffer->TransitionImageLayout(commandBuffer, vulkanSwapChain->GetColorImage(swapChainBufferIndex), *vulkanSwapChain->GetColorFormat(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = vulkanSwapChain->GetColorImageView(swapChainBufferIndex),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clearColor
    };
    VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = vulkanSwapChain->GetDepthImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.depthStencil = {1.0f, 0}}
    };
    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment,
    };

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
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

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mesh->indices.size()), 1, 0, 0, 0);

    vkCmdEndRendering(commandBuffer);

    vulkanCommandBuffer->TransitionImageLayout(commandBuffer, vulkanSwapChain->GetColorImage(swapChainBufferIndex), *vulkanSwapChain->GetColorFormat(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void VulkanApp::recordCommandBuffer(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkRenderPass renderPass, uint32_t swapChainBufferIndex, VkExtent2D swapChainExtent)
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