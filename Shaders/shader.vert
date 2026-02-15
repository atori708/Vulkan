#version 450

layout(set = 0, binding = 0) uniform CameraUniformBufferObject {
    mat4 view;
    mat4 proj;
} cameraUbo;

layout(set = 1, binding = 0) uniform ModelUniformBufferObject {
    mat4 model;
    vec3 color;
} modelUbo;

layout(location = 0) in vec3 inPosition; // 入力頂点座標
layout(location = 1) in vec3 inColor; // 入力頂点カラー
layout(location = 2) in vec2 inTexCoord; // 入力UV座標

layout(location = 0) out vec3 fragColor; // 出力頂点カラー
layout(location = 1) out vec2 fragTexCoord; // 出力UV座標

void main() {
    gl_Position = cameraUbo.proj * cameraUbo.view * modelUbo.model * vec4(inPosition, 1.0);
    fragColor = modelUbo.color;
    fragTexCoord = inTexCoord;
}