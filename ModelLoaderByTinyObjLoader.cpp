#include "ModelLoaderByTinyObjLoader.h"
#include <stdexcept>

/// <summary>
/// Objモデルの読み込み
/// </summary>
Mesh* ModelLoaderByTinyObjLoader::loadModel(const std::string& modelPath)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &err, modelPath.c_str())) {
        throw std::runtime_error(err);
    }

    Mesh* mesh = new Mesh();

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0 - attrib.texcoords[2 * index.texcoord_index + 1] // OBJのテクスチャ座標は左下が原点なので反転する
            };

            vertex.color = { 1.0f, 1.0f, 1.0f };

            mesh->vertices.push_back(vertex);
            mesh->indices.push_back(static_cast<uint32_t>(mesh->indices.size()));

            // TODO Tutorialでは重複頂点の対応があるのであとでやる
        }
    }

    return mesh;
};