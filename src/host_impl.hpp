#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"

namespace nn
{
class HostImpl
{
public:
    ConstTensor<> forward(NetworkData& data)
	{
        const size_t inputSize = data.inputShape.size();
        const size_t layerCount = data.layers.size();
        const float* layerInput = data.inputData.data();
        float* layerOutput = layerOutputs.data();

        for (size_t i = 0; i < inputSize; ++i)
        {
            for (size_t j = 0; j < layerCount; ++j)
            {
                data.layers[j]->forward(layerInput, layerOutput);
                layerInput = layerOutput;
                layerOutput += data.layers[j]->getOutputShape().size();
            }
        }

        const size_t outputSize = data.layers.back()->getOutputShape().size();
        return ConstTensor<>(outputSize, &layerOutputs.back() - outputSize);
	}

    void train(NetworkData& data)
    {

    }


private:
    vector<float> layerOutputs;
};
}