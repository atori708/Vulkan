#pragma once
#include <vector>
#include"MeshFormat.h"

class Mesh
{
public:
    ~Mesh();
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
};

