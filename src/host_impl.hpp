#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"
#include "losses/loss.hpp"
#include "impl.hpp"
#include "../utils/utils.hpp"

namespace nn
{
class HostImpl : public Impl
{
public:
	HostImpl(unique_ptr<NetworkConfig>&& config) :
		Impl(move(config))
	{
		size_t totalOutputs = 0;
		size_t parameterCount = 0;

		for (const auto& layer : getConfig().layers)
		{
			totalOutputs += layer->getOutputSize();
			parameterCount += layer->getParameterCount();
		}

		layerOutputs = Tensor<>(totalOutputs);
		layerError = Tensor<>(totalOutputs);
		parameters = Tensor<>(parameterCount);
		optimizerData = Tensor<>(getConfig().optimizer->getRequiredSize(parameterCount));

		Impl::parameters = parameters.data();
		Impl::optimizerData = optimizerData.data();
		Impl::layerOutputs.push_back(layerOutputs.data());
		Impl::layerError.push_back(layerError.data());
		Impl::layerParams.push_back(parameters.data());
		Impl::layerDerivatives.push_back(optimizerData.data());
		getConfig().layers[0]->initializeParameters(Impl::layerParams.back());

		for (size_t i = 1; i < getConfig().layers.size(); ++i)
		{
			auto& layer = *getConfig().layers[i];
			Impl::layerOutputs.push_back(Impl::layerOutputs.back() + layer.getInputSize());
			Impl::layerError.push_back(Impl::layerError.back() + layer.getInputSize());
			Impl::layerParams.push_back(parameters.data() + layer.getParameterCount());
			Impl::layerDerivatives.push_back(optimizerData.data() + layer.getParameterCount());
			layer.initializeParameters(Impl::layerParams.back());
		}
		
		getConfig().optimizer->init(parameters.data(), parameterCount);
	}

private:
	void forward(layer::Layer& layer, const float* input, const float* params, float* output) const final
	{
		layer.forward(input, params, output);
	}

	void calculateOutputDerivatives(const float* output, const uint32_t* target, size_t outputSize, float* outputError) const final
	{
		memcpy(outputError, output, outputSize * sizeof(float));
		outputError[*target] -= 1.0f;
	}

	void calculateOutputDerivatives(const float* output, const float* target, size_t outputSize, float* outputError) const
	{
		getConfig().lossFunc->calculateDerivatives(output, target, outputError, outputSize);
	}

	void calculateLoss(const float* output, const uint32_t* target, size_t outputSize, uint32_t* result) const final
	{
		*result = argMax(output, outputSize) == *target;
	}

	void calculateLoss(const float* output, const float* target, size_t outputSize, float* result) const final
	{
		*result = getConfig().lossFunc->calculateError(output, target, outputSize);
	}

	void backPropagate(layer::Layer& layer, const layer::Layer::BackPropData& data, float* inputError) const final
	{
		layer.backPropagate(data, inputError);
	}

	void calculateDerivatives(layer::Layer& layer, const layer::Layer::BackPropData& data, float* derivatives) const final
	{
		layer.calculateDerivatives(data, derivatives);
	}
	// outputs of each layer
	Tensor<> layerOutputs;

	// error of each layer output during backpropagation
	Tensor<> layerError;

	// network weights and biases
	Tensor<> parameters;

	// data used by optimiser (e.g. derivatives)
	Tensor<> optimizerData;
};
}