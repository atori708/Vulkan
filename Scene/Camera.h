#pragma once
#include <glm/glm.hpp>

struct Camera {
    glm::vec3 position;
    glm::vec3 lookat;
    glm::vec3 up;
    float fov;
    float aspect;
    float nearClip;
    float farClip;
};