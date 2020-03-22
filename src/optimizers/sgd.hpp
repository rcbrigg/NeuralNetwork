#pragma once
#include "optimizer.hpp"

namespace nn
{
namespace optimizer
{
class Sgd : public Optimizer
{
public:
	Sgd(float learningRate) :
		learningRate(learningRate)
	{
	}

	Tensor<> getDerivatives() final
	{
		return derivatives;
	}

	void update(float* params, size_t batchSize) final
	{
		const size_t size = derivatives.size();
		const float scale = learningRate / batchSize;
		const float* derivative = derivatives.data();

		for (size_t i = 0; i < size; ++i)
		{
			params[i] -= scale * derivative[i];
		}
	}

private:

	float learningRate;

	Tensor<> derivatives;
};
}
}
