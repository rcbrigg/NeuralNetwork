#pragma once
#include <stdint.h>
#include "..\include\tensor.hpp"
#include "..\include\shape.hpp"
#include <memory>
#include <vector>
using namespace std;

namespace nn
{
namespace Layer { class Layer; }

class Optimizer;

struct NetworkData
{
    bool compiled = false;

    uint32_t batchSize = 0;

    Tensor<> inputData;

    Tensor<> m_targetData;

    Tensor<1, uint32_t> m_targetLabels;

    Shape inputShape;

    unique_ptr<Optimizer> m_lossFunction;

    unique_ptr<Optimizer> m_optimier;

    vector<unique_ptr<layer::Layer>> layers;
};

}
