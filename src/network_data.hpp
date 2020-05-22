#pragma once
#include <stdint.h>
#include "tensor.hpp"
#include "shape.hpp"
#include <memory>
#include <vector>
using namespace std;

namespace nn
{
namespace layer { class Layer; }
namespace optimizer { class Optimizer; }
namespace loss { class Loss; }


struct NetworkConfig
{
    uint32_t batchSize = 1;

    bool cl = false;

    Shape<> inputShape;

    Shape<> outputShape;

    unique_ptr<loss::Loss> lossFunc;

    unique_ptr<optimizer::Optimizer> optimizer;

    vector<unique_ptr<layer::Layer>> layers;
};

}
