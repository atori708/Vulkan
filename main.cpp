#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <stdexcept>

#include "VulkanApp.h"

class HelloTriangleApplication
{
public:
    void run() {
        initWindow();
        vulkanApp = new VulkanApp(window);
        mainLoop();
        cleanup();
    }

private:
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    GLFWwindow* window;
    bool frameBufferResized = false;
    uint32_t currentFrame = 0;

    VulkanApp* vulkanApp;

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
        app->OnWindowResized(width, height);
    }

    void OnWindowResized(int width, int height)
    {
        vulkanApp->OnWindowResized();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            vulkanApp->Draw(currentFrame);
            currentFrame++;
        }
    }

    void cleanup()
    {
        delete vulkanApp;
        vulkanApp = nullptr;

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
