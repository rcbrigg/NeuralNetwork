#pragma once

namespace nn
{
namespace loss
{
class Loss
{
public:
	virtual void calculateError(const float* output, const float* target, float* error, size_t size) = 0;

	virtual float calculateError(const float* output, const float* target, size_t size) = 0;

	virtual void calculateDerivatives(const float* output, const float* target, float* derivatives, size_t size) = 0;
};
}
}
