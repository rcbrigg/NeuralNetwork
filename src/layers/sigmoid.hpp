#pragma once
#include "layer.hpp"
#include <math.h>
#include <cstring>

namespace nn
{
namespace layer
{
class Sigmoid : public Layer
{
public:
	Sigmoid(size_t size) :
		Layer(size, size, 0)
	{
	}

	void forward(const float* input, const float*, float* output) const final
	{
		for (size_t i = 0; i < inputSize; ++i)
		{
			output[i] = sigmoid(input[i]);
		}
	}

	void backPropagate(const BackPropData& data, float* inputError) const final
	{
		for (size_t i = 0; i < inputSize; ++i)
		{
			inputError[i] = sigmoidPrime(data.input[i]) * data.outputError[i];
		}
	}

	static float sigmoid(float x)
	{
		return 1.f / (1.f + expf(-x));
	}

	static float sigmoidPrime(float x)
	{
		const float s = sigmoid(x);
		return s * (1.f - s);
	}
};
}
}