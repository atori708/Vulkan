#pragma once
#include <string>
#include "Mesh.h"

class IModelLoader
{
public:
    virtual Mesh* loadModel(const std::string& modelPath) = 0;
};

