#pragma once
#include "tensor.hpp"
#include "shape.hpp"
#include "dimension.hpp"

namespace nn
{
enum class LossFunction
{
    None,
    MeanSquare
};

enum class OptimizerType
{
    None,
    GranientDescent,
    Adam
};

enum class LayerType
{
    None,
    Dense,
    Canv1d,
    Conv2d,
    Conv3d,
    MaxPooling,
    Dropout,
    Sigmoid,
    Tanh,
    Relu,
    Lrelu
};

class Network
{
public:
    Network();

    ~Network();

    template<size_t N> void setInputSize(Shape shape);

    template<size_t N> void setInputData(const Tensor<N>& data);

    void addLayerDense(uint32_t outputSize);

    void addLayerConv2d(uint32_t fileterCount, Dim2 filterSize, Dim2 filterStep = { 1, 1 });

    void addLayerMaxPooling(Dim2 poolingSize);

    void addLayerSigmoid();

    void addLayerTanh();

    void addLayerRelu();

    void addLayerLrelu();

    void setOptimizerGradientDescent(float learningRate = 0.01);

    void setOptimizerAdam();

    void batchSize(uint32_t size);

    void compile(bool gpu = true);

    template<size_t N> void setTargets(const Tensor<>& data);

    void setLabels(const Tensor<>& data);

    void train(uint32_t epochs = 1);

    Tensor<> forward();

    Tensor<1, uint32_t> classify();

private:
    class Impl;
    Impl* impl;
};
}