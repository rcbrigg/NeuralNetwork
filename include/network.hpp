#pragma once
#include "tensor.hpp"
#include "shape.hpp"
#include "dimension.hpp"

#include <memory>

namespace nn
{

class Network
{
public:
    Network(class NetworkArgs&& args);

    Network(Network&&) = default;

    ~Network();

	template<size_t N, typename T> Tensor<> forward(const Tensor<N, T>& inputs)
	{
        static_assert(N > 1, "Expected input for forward() to have at least 2 dimensions. Note: can use Tensor::as({1, n})");
        checkInputShape(inputs.shape());
		return forward(inputs.flat(), inputs.length());
	}

    template<size_t N, typename T> Tensor<1, uint32_t> clasify(const Tensor<N, T>& inputs)
    {
        static_assert(N > 1, "Expected input for clasify() to have at least 2 dimensions. Note: can use Tensor::as({1, n})");
        checkInputShape(inputs.shape());
        return clasify(inputs.flat(), inputs.length());
    }

    // For normal training. U should be float or const float.
	template<size_t N, typename T, typename U> void train(const Tensor<N, T> inputs, const Tensor<2, U> targets, uint32_t epochs = 1)
	{		
        static_assert(N > 1, "Expected input for train() to have at least 2 dimensions. Note: can use Tensor::as({1, n})");
        checkTargetShape(inputs.shape(), targets.shape());
		train(inputs.flat(), targets.flat(), inputs.length(), epochs);
	}

    // For classification training. U should be uint32_t or const uint32_t.
    template<size_t N, typename T, typename U> void train(const Tensor<N, T> inputs, const Tensor<1, U> targets, uint32_t epochs = 1)
    {
        static_assert(N > 1, "Expected input for train() to have at least 2 dimensions. Note: can use Tensor::as({1, n})");
        checkLabels(inputs.shape(), targets.shape());
        train(inputs.flat(), targets.flat(), inputs.length(), epochs);
    }

    template<size_t N, typename T, typename U> double test(const Tensor<N, T>& inputs, const Tensor<2, U>& targets)
    {
        static_assert(N > 1, "Expected input for test() to have at least 2 dimensions. Note: can use Tensor::as({1, n})");
        checkTargetShape(inputs.shape(), targets.shape());
        return test(inputs.flat(), targets.flat(), inputs.length());
    }

    template<size_t N, typename T, typename U> double test(const Tensor<N, T>& inputs, const Tensor<1, U>& targets)
    {
        static_assert(N > 1, "Expected input for test() to have at least 2 dimensions. Note: can use Tensor::as({1, n})");
        checkLabels(inputs.shape(), targets.shape());
        return test(inputs.flat(), targets.flat(), inputs.length());
    }
private:

	Tensor<> forward(ConstTensor<> inputs, size_t inputCount);
    Tensor<1, uint32_t> clasify(ConstTensor<> inputs, size_t inputCount);
	void train(ConstTensor<> inputs, ConstTensor<> targets, size_t inputCount, uint32_t epochs);
    void train(ConstTensor<> inputs, Tensor<1, const uint32_t> targets, size_t inputCount, uint32_t epochs);
    double test(ConstTensor<> inputs, ConstTensor<> targets, size_t inputCount);
    double test(ConstTensor<> inputs, Tensor<1, const uint32_t> targets, size_t inputCount);
	void checkInputShape(Shape<> inputShape) const;
	void checkTargetShape(Shape<> inputShape, Shape<> targetShape) const;
    void checkLabels(Shape<> inputShape, Shape<> targetShape) const;
    void checkLossFunction() const;
    void checkOptimizer() const;

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