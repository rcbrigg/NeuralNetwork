#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"
#include "../utils/utils.hpp"
#include "losses/loss.hpp"
#include "impl.hpp"

namespace nn
{
class HostImpl : public Impl
{
public:
	HostImpl(unique_ptr<const NetworkConfig>&& config) : Impl(move(config))
	{
	}

	bool init() final
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
		optimizerData = Tensor<>(parameterCount);

		float* layerParams = parameters.data();
		for (const auto& layer : config->layers)
		{
			layer->initializeParameters(layerParams);
			layerParams += layer->getParameterCount();
		}

		if (config->optimizer)
		{
			config->optimizer->init(parameters.data(), parameterCount);
		}
		return true;
	}

	Tensor<> forward(const ConstTensor<>& input, size_t inputCount) final
	{
		const size_t outputSize = config->layers.back()->getOutputSize();
		const size_t inputSize = config->inputShape.size();
		const size_t requiredSize = inputCount * outputSize;

		outputs = Tensor<>(requiredSize);

		float* layerOutput = outputs.data();
		const float* layerInput = input.data();

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(config->layers, layerInput, parameters.data(), layerOutputs.data(), layerOutput);
			layerInput += inputSize;
			layerOutput += outputSize;
		}

		return outputs;
	}

	Tensor<1, uint32_t> clasify(const ConstTensor<>& input, size_t inputCount) final
	{
		const size_t outputSize = config->layers.back()->getOutputSize();
		const size_t inputSize = config->inputShape.size();

		classifications = Tensor<1, uint32_t>(inputCount);

		float* networkOutput = layerOutputs.data(-ptrdiff_t(outputSize));
		const float* layerInput = input.data();

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(config->layers, layerInput, parameters.data(), layerOutputs.data(), networkOutput);
			classifications[i] = argMax(networkOutput, outputSize);
		}

		return classifications;
	}

	double test(const ConstTensor<>& input, const Tensor<1, const uint32_t>& targets, size_t inputCount) final
	{
		return testCommon(input, targets, inputCount);
	}

	double test(const ConstTensor<>& input, const Tensor<1, const float>& targets, size_t inputCount) final
	{
		return testCommon(input, targets, inputCount);
	}

	template<typename T>
	double testCommon(const ConstTensor<>& input, const Tensor<1, const T>& targets, size_t inputCount)
	{
		double loss = 0.0;

		const size_t outputSize = config->layers.back()->getOutputSize();
		const size_t inputSize = config->inputShape.size();
		const size_t targetSize = getTargetSize<T>();

		float* networkOutput = layerOutputs.data(-ptrdiff_t(outputSize));
		const float* layerInput = input.data();
		const T* target = targets.data();

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(config->layers, layerInput, parameters.data(), layerOutputs.data(), networkOutput);
			loss += calculateLoss(networkOutput, target, outputSize);
			layerInput += inputSize;
			target += targetSize;
		}

		return loss / double(inputCount);
	}

	void train(const ConstTensor<>& inputs, const Tensor<1, const float>& targets, size_t inputCount) final
	{
		trainCommon(inputs, targets, inputCount);
	}

	void train(const ConstTensor<>& inputs, const Tensor<1, const uint32_t>& targets, size_t inputCount) final
	{
		trainCommon(inputs, targets, inputCount);
	}

	template<typename T>
	void trainCommon(const ConstTensor<>& inputs, const Tensor<1, const T>& targets, size_t inputCount)
	{
		float* derivatives = this->optimizerData.data();

		const size_t inputSize = config->inputShape.size();
		const size_t targetSize = getTargetSize<T>();

		const T* target = targets.data();
		const float* input = inputs.data();

		for (size_t i = 0 ; i < inputCount;)
		{
			size_t batchEnd = std::min(i + config->batchSize, inputCount);
			size_t batchSize = batchEnd - i;
			config->optimizer->beginBatch(optimizerData.data());

			for (; i < batchEnd; ++i)
			{
				train(input, target);
				input += inputSize;
				target += targetSize;
			}

			config->optimizer->update(parameters.data(), optimizerData.data(), batchSize);
		}	
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
			layerOutput += layers[i]->getOutputSize();
			layerParams += layers[i]->getParameterCount();
		}

		layers.back()->forward(layerInput, layerParams, output);
	}

	template<typename T>
	void train(const float* input, T target)
	{
		size_t outputSize = config->outputShape.size();

		forward(config->layers,
				input,
				parameters.data(),
				layerOutputs.data(),
				layerOutputs.data(-(ptrdiff_t)outputSize));

		const size_t layerCount = config->layers.size();
		const size_t layerParamCount = config->layers.back()->getParameterCount();
		float* outputError = layerError.data(-(ptrdiff_t)outputSize);

		layer::Layer::BackPropData data;
		data.params = parameters.end();
		data.output = layerOutputs.data(-(ptrdiff_t)outputSize);
		data.outputError = outputError;

		calculateOutputDerivatives(data.output, target, outputSize, outputError);

		float* derivatives = optimizerData.data() + parameters.size();
		const float* layerParams = parameters.data(config->layers.back()->getParameterCount());

		for (size_t i = layerCount - 1; i > 0; --i)
		{
			layer::Layer& layer = *config->layers[i];

			float* inputError = outputError - layer.getInputSize();
			data.input = data.output - layer.getInputSize();
			data.params -= layer.getParameterCount();
			derivatives -= layer.getParameterCount();

			layer.backPropagate(data, inputError);
			layer.calculateDerivatives(data, derivatives);

			outputError = inputError;
			data.outputError = outputError;
			data.output = data.input;
		}

		data.input = input;
		data.params -= config->layers[0]->getParameterCount();
		derivatives -= config->layers[0]->getParameterCount();
		config->layers[0]->calculateDerivatives(data, derivatives);
	}

	void calculateOutputDerivatives(const float* output, const uint32_t* target, size_t outputSize, float* outputError) const
	{
		memcpy(outputError, output, outputSize * sizeof(float));
		outputError[*target] -= 1.0f;
	}

	void calculateOutputDerivatives(const float* output, const float* target, size_t outputSize, float* outputError) const
	{
		return config->lossFunc->calculateDerivatives(output, target, outputError, outputSize);
	}

	auto calculateLoss(const float* output, const uint32_t* target, size_t outputSize) const
	{
		return argMax(output, outputSize) == *target;
	}

	auto calculateLoss(const float* output, const float* target, size_t outputSize) const
	{
		return config->lossFunc->calculateError(output, target, outputSize);
	}

	template<typename T> size_t getTargetSize() { return config->layers.back()->getOutputSize(); }

	template<> size_t getTargetSize<uint32_t>() { return 1; }

	// outputs of each layer
	Tensor<> layerOutputs;

	// error of each layer output during backpropagation
	Tensor<> layerError;

	// outputs of final layer
	Tensor<> outputs;

	// maximum output of final layer
	Tensor<1, uint32_t> classifications;

	// network weights and biases
	Tensor<> parameters;

	// data used by optimiser (e.g. derivatives)
	Tensor<> optimizerData;
};
}