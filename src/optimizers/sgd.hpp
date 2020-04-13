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
		learningRate(learningRate),
		parameterCount(0)
	{
	}

	float* getDerivatives(float* data) const final
	{
		return data;
	}

	size_t getRequiredSize(size_t paramCount) const final
	{
		return paramCount;
	}

	void init(float* data, size_t paramCount) final
	{
		parameterCount = paramCount;
	}

	void beginBatch(float* data) final
	{
		memset(data, 0, sizeof(float) * parameterCount);
	}

	void update(float* params, const float* data, size_t batchSize) final
	{
		const size_t size = parameterCount;
		const float scale = learningRate / batchSize;
		const float* derivative = data;

		for (size_t i = 0; i < parameterCount; ++i)
		{
			params[i] -= scale * derivative[i];
		}
	}

private:

	float learningRate;

	size_t parameterCount;
};
}
}
