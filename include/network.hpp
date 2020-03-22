#pragma once
#include "tensor.hpp"
#include "shape.hpp"
#include "dimension.hpp"

#include <memory>

namespace nn
{
//enum class LossFunction
//{
//    None,
//    MeanSquare
//};
//
//enum class OptimizerType
//{
//    None,
//    GranientDescent,
//    Adam
//};
//
//enum class LayerType
//{
//    None,
//    Dense,
//    Canv1d,
//    Conv2d,
//    Conv3d,
//    MaxPooling,
//    Dropout,
//    Sigmoid,
//    Tanh,
//    Relu,
//    Lrelu
//};

class Network
{
public:
    Network(class NetworkArgs&& args);

    Network(Network&&) = default;

    ~Network();

	template<size_t N, typename T> Tensor<> forward(const Tensor<N, T>& inputs)
	{
        checkInputShape(inputs.shape());
		return forward(inputs.flat(), inputs.length());
	}

	template<size_t N, typename T, typename U> void train(const Tensor<N, T>& inputs, const Tensor<2, U>& targets, uint32_t epochs)
	{		
        checkTrainingShape(inputs.shape(), targets.shape());
		train(inputs.flat(), targets.flat(), inputs.length());
	}

private:
	Tensor<> forward(ConstTensor<> inputs, size_t inputCount);

	void train(ConstTensor<> inputs, ConstTensor<> targets, size_t inputCount);

	void checkInputShape(Shape<> inputShape);
	void checkTrainingShape(Shape<> inputShape, Shape<> targetShape);

    std::unique_ptr<class HostImpl> impl;
};

class NetworkArgs
{
public:
    NetworkArgs();

    ~NetworkArgs();

    void setInputShape(Shape<> shape);

    void addLayerDense(uint32_t outputSize);

    void addLayerConv2d(uint32_t fileterCount, Shape<2> filterSize, Shape<2> filterStep = Shape<2>({ 1, 1 }));

    void addLayerMaxPooling(Dim2 poolingSize);

    void addLayerSigmoid();

    void addLayerTanh();

    void addLayerRelu();

    void addLayerLrelu();

    void setOptimizerGradientDescent(float learningRate = 0.01);

    void setOptimizerAdam();

    void setLossMse();

    void setBatchSize(uint32_t size);

    Shape<>& getOutputShape() const;

private:
    void checkAddLayer();

    friend Network;

    std::unique_ptr<struct NetworkConfig> data;
};


}