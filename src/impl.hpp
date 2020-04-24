#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"
#include "../utils/utils.hpp"
#include "losses/loss.hpp"
#include "common.hpp"

namespace nn
{
class Impl
{
public:
	Impl(unique_ptr<NetworkConfig>&& config)
	{
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

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(layerInput, layerOutput);
			layerInput += inputSize;
			layerOutput += outputSize;
		}

		return outputs;
	}

	Tensor<1, uint32_t> clasify(const ConstTensor<>& input, size_t inputCount)
	{
		const size_t outputSize = config->layers.back()->getOutputSize();
		classifications = Tensor<1, uint32_t>(inputCount);

		float* networkOutput = layerOutputs.back(); 
		const float* layerInput = input.data();

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(layerInput, networkOutput);
			classifications[i] = argMax(networkOutput, outputSize);
		}

		return classifications;
	}

	template<typename T>
	double test(const ConstTensor<>& input, const Tensor<1, const T>& targets, size_t inputCount)
	{
		double totalLoss = 0.0;

		const size_t outputSize = config->layers.back()->getOutputSize();
		const size_t inputSize = config->inputShape.size();
		const size_t targetSize = getTargetSize<T>();

		float* networkOutput = layerOutputs.back();
		const float* layerInput = input.data();
		const T* target = targets.data();

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(layerInput, networkOutput);
			T loss;
			calculateLoss(networkOutput, target, outputSize, &loss);
			totalLoss += loss;
			layerInput += inputSize;
			target += targetSize;
		}

		return totalLoss / double(inputCount);
	}

	template<typename T>
	void train(const ConstTensor<>& inputs, const Tensor<1, const T>& targets, size_t inputCount)
	{
		const size_t inputSize = config->inputShape.size();
		const size_t targetSize = getTargetSize<T>();

		const T* target = targets.data();
		const float* input = inputs.data();

		for (size_t i = 0; i < inputCount;)
		{
			size_t batchEnd = std::min(i + config->batchSize, inputCount);
			size_t batchSize = batchEnd - i;
			config->optimizer->beginBatch(optimizerData);

			for (; i < batchEnd; ++i)
			{
				train(input, target);
				input += inputSize;
				target += targetSize;
			}

			config->optimizer->update(parameters, optimizerData, batchSize);
		}
	}

	const NetworkConfig& getConfig() const
	{
		return *config;
	}

private:
	using Layers = vector<unique_ptr<layer::Layer>>;

	virtual void forward(layer::Layer& layer, const float* input, const float* params, float* output) const = 0;
	virtual void backPropagate(layer::Layer& layer, const layer::Layer::BackPropData& data, float* inputError) const = 0;
	virtual void calculateDerivatives(layer::Layer& layer, const layer::Layer::BackPropData& data, float* derivatives) const = 0;

	void forward(const float* input, float* output) const
	{
		const size_t layerCount = config->layers.size();
		const float* layerInput = input;

		for (size_t i = 0; i < layerCount - 1; ++i)
		{
			forward(*config->layers[i], layerInput, layerParams[i], layerOutputs[i]);
			layerInput = layerOutputs[i];
		}

		config->layers.back()->forward(layerInput, layerParams.back(), output);
	}

	template<typename T>
	void train(const float* input, T target)
	{
		forward(input, layerOutputs.back());

		const size_t layerCount = config->layers.size();
		float* outputError = layerError.back();

		layer::Layer::BackPropData data;
		data.output = layerOutputs.back();
		data.outputError = outputError;

		calculateOutputDerivatives(data.output, target, config->outputShape.size(), outputError);

		for (size_t i = layerCount - 1; i > 0; --i)
		{
			layer::Layer& layer = *config->layers[i];

			float* inputError = layerError[i - 1];
			data.input = layerOutputs[i-1];
			data.params = layerParams[i];

			layer.backPropagate(data, inputError);
			layer.calculateDerivatives(data, layerDerivatives[i]);

			outputError = inputError;
			data.outputError = layerError[i - 1];;
			data.output = data.input;
		}

		data.input = input;
		data.params = layerParams[0];
		config->layers[0]->calculateDerivatives(data, layerDerivatives[0]);
	}

	virtual void calculateOutputDerivatives(const float* output, const uint32_t* target, size_t outputSize, float* outputError) const = 0;

	virtual void calculateOutputDerivatives(const float* output, const float* target, size_t outputSize, float* outputError) const = 0;

	virtual void calculateLoss(const float* output, const uint32_t* target, size_t outputSize, uint32_t* result) const = 0;

	virtual void calculateLoss(const float* output, const float* target, size_t outputSize, float* result) const = 0;

	template<typename T> size_t getTargetSize() { return config->layers.back()->getOutputSize(); }

	template<> size_t getTargetSize<uint32_t>() { return 1; }

protected:

	// outputs of each layer
	std::vector<float*> layerOutputs;

	// error of each layer output during backpropagation
	std::vector<float*> layerError;
	
	// network weights and biases
	float* parameters;

	// network weights and biases for each layer
	std::vector<float*> layerParams;

	// data used by optimiser (e.g. derivatives)
	float* optimizerData;

	// parameter gradients for each layer
	std::vector<float*> layerDerivatives;

private:

	// outputs of final layer
	Tensor<> outputs;

	// maximum output of final layer
	Tensor<1, uint32_t> classifications;

	unique_ptr<NetworkConfig> config;
};
}