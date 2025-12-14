#pragma once
#include "IModelLoader.h"
#include <tiny_obj_loader.h>

class ModelLoaderByTinyObjLoader : public IModelLoader
{
public:
    Mesh* loadModel(const std::string& modelPath) override;
};

