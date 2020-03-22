#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"
#include "losses/loss.hpp"

namespace nn
{
class HostImpl
{
public:
	HostImpl(unique_ptr<NetworkConfig>&& config)
	{
		size_t totalOutputs = 0;
		size_t parameterCount = 0;

		for (const auto& layer : config->layers)
		{
			totalOutputs += layer->getOutputSize();
			parameterCount += layer->getParameterCount();
		}

		layerOutputs = Tensor<>(totalOutputs);
		layerError = Tensor<>(totalOutputs);
		parameters = Tensor<>(parameterCount);

		this->config = move(config);
	}

	Tensor<> forward(const ConstTensor<>& input, size_t inputCount)
	{
		const size_t outputSize = config->layers.back()->getOutputSize();
		const size_t inputSize = config->inputShape.size();
		const size_t requiredSize = inputCount * outputSize;

		outputs = Tensor<>(requiredSize);

		float* layerOutput = outputs.data();
		const float* layerInput = input.data();

		for (size_t i = 1; i < inputCount; ++i)
		{
			forward(config->layers, layerInput, config->parameters.data(), layerOutputs.data(), layerOutput);
			layerInput += inputSize;
			layerOutput += outputSize;
		}

		return outputs;
	}

	void train(const ConstTensor<>& inputs, const ConstTensor<>& targets, size_t inputCount)
	{
		const size_t inputSize = config->inputShape.size();
		const size_t outputSize = config->outputShape.size();

		const float* target = targets.data();
		const float* input = inputs.data();

		for (size_t i = 0 ; i < inputCount; ++i)
		{
			train(input, target);
			input += inputSize;
			target += outputSize;
		}
	}

	void compile(NetworkConfig& data)
	{
		size_t outputSize = 0;
		size_t paramSize = 0;

	}

	const NetworkConfig& getConfig() const
	{
		return *config;
	}
private:
	using Layers = vector<unique_ptr<layer::Layer>>;

	void forward(const Layers& layers,
				 const float* input,
				 const float* parameters,
				 float* layerOutputs,
				 float* output) const
	{
		const size_t layerCount = layers.size();
		const float* layerInput = input;
		const float* layerParams = parameters;
		float* layerOutput = layerOutputs;

		for (size_t i = 0; i < layerCount - 1; ++i)
		{
			layers[i]->forward(layerInput, layerParams, layerOutput);
			layerInput = layerOutput;
			layerParams += layers[i]->getParameterCount();
		}

		layers.back()->forward(layerInput, layerParams, output);
	}

	void train(const float* input,
			   const float* target)
	{
		size_t outputSize = config->outputShape.size();

		forward(config->layers,
				input, parameters.data(),
				layerOutputs.data(),
				layerOutputs.data(-(ptrdiff_t)outputSize));

		const size_t layerCount = config->layers.size();
		const size_t layerParamCount = config->layers.back()->getParameterCount();
		float* outputError = layerError.data(-(ptrdiff_t)outputSize);

		layer::Layer::BackPropData data;
		data.params = config->parameters.data(-(ptrdiff_t)layerParamCount);
		data.output = layerOutputs.data(-(ptrdiff_t)outputSize);
		data.outputError = outputError;

		config->lossFunc->calculateError(data.output, target, outputError, outputSize);

		float* derivatives = config->optimizer->getDerivatives().data(-(ptrdiff_t)layerParamCount);
		const float* layerParams = config->parameters.data(config->layers.back()->getParameterCount());

		for (size_t i = layerCount - 1; i > 0; --i)
		{
			layer::Layer& layer = *config->layers[i];

			float* inputError = outputError - layer.getOutputSize();
			data.input = data.output - layer.getInputSize();
			data.params -= layer.getParameterCount();
			
			layer.backPropagate(data, inputError);
			layer.calculateDerivatives(data, derivatives);

			outputError = inputError;
			data.outputError = outputError;
			data.output = data.input;
		}

		data.input = input;
		data.params = parameters.data();
		config->layers[0]->calculateDerivatives(data, derivatives);
	}

	// outputs of each layer
	Tensor<> layerOutputs;

	// error of each layer output during backpropagation
	Tensor<> layerError;

	// outputs of final layer
	Tensor<> outputs;

	// network weights and biases
	Tensor<> parameters;

	unique_ptr<NetworkConfig> config;
};
}