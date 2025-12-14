#pragma once
#include"IModelLoader.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class ModelLoaderAssimp : public IModelLoader
{
public:
    Mesh* loadModel(const std::string& modelPath) override;
};
