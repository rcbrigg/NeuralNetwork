#pragma once
#include "loss.hpp"

namespace nn
{
namespace loss
{
class Mse : public Loss
{
public:

	void calculateError(const float* output, const float* target, float* error, size_t size) final
	{
		for (size_t i = 0; i < size; ++i)
		{
			error[i] = square(output[i] - target[i]);
		}
	}

	void calculateDerivatives(const float* output, const float* target, float* derivatives, size_t size) final
	{
		for (size_t i = 0; i < size; ++i)
		{
			derivatives[i] = 2.f * (output[i] - target[i]);
		}
	}

private:
	static float square(float x) { return x * x; }
};
}
}
