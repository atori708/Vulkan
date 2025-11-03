#include "ModelLoaderAssimp.h"
#include <stdexcept>

Mesh* ModelLoaderAssimp::loadModel(const std::string& modelPath)
{
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(modelPath, 
        aiProcess_Triangulate | 
        aiProcess_FlipUVs | 
        aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error("Assimp failed to load model: " + std::string(importer.GetErrorString()));
    }

    Mesh* mesh = new Mesh();

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* aiMesh = scene->mMeshes[i];

        for (unsigned int j = 0; j < aiMesh->mNumVertices; j++) {
            Vertex vertex{};

            vertex.pos = {
                aiMesh->mVertices[j].x,
                aiMesh->mVertices[j].y,
                aiMesh->mVertices[j].z
            };

            if (aiMesh->mTextureCoords[0]) {
                vertex.texCoord = {
                    aiMesh->mTextureCoords[0][j].x,
                    aiMesh->mTextureCoords[0][j].y
                };
            } else {
                vertex.texCoord = { 0.0f, 0.0f };
            }

            vertex.color = { 1.0f, 1.0f, 1.0f };

            mesh->vertices.push_back(vertex);
        }

        for (unsigned int j = 0; j < aiMesh->mNumFaces; j++) {
            aiFace face = aiMesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                mesh->indices.push_back(face.mIndices[k]);
            }
        }
    }

    return mesh;
}
