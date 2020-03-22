#pragma once
#include "layer.hpp"
#include <cstring>

namespace nn
{
namespace layer
{
class Dense : public Layer
{
public:
	Dense(size_t inputSize, size_t outputSize) :
		Layer(inputSize, outputSize, inputSize * (1 + outputSize))
	{
	}

	void forward(const float* input, const float* params, float* output) const final
	{
		const float* bias = params;
		const float* weight = params + outputSize;
		
		for (size_t i = 0; i < outputSize; ++i)
		{
			output[i] = *bias++;
			for (size_t j = 0; j < inputSize; ++j)
			{
				output[i] += input[j] * *weight++;
			}
		}
	}

	void backPropagate(const BackPropData& data, float* inputError) const final
	{
		memset(inputError, 0, inputSize * sizeof(float));

		const float* weight = data.params + outputSize;

		for (size_t i = 0; i < outputSize; ++i)
		{
			for (size_t j = 0; j < inputSize; ++j)
			{
				inputError[j] += data.outputError[i] * *weight++;
			}
		}
	}

	void calculateDerivatives(const BackPropData& data, float* derivatives) const final
	{
		float* db = derivatives;
		float* dw = derivatives + outputSize;

		for (size_t i = 0; i < outputSize; ++i)
		{
			*db++ += data.outputError[i];
			for (size_t j = 0; j < inputSize; ++j)
			{
				*dw++ += data.outputError[i] * data.input[j];
			}
		}
	}
};
}
}