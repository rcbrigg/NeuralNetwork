#pragma once
#include "tensor.hpp"

namespace nn
{
namespace optimizer
{
class Optimizer
{
public:
	// Get pointer to derivitive data
	virtual float* getDerivatives(float* data) const = 0;

	// Get number of float parameters required by this optimizer
	virtual size_t getRequiredSize(size_t paramCount) const = 0;

	// Perform any necessary data initialization
	virtual void init(float* data, size_t paramCount) = 0;

	// Called at the start of each batch
	virtual void beginBatch(float* data) = 0;

	// Update parameters after backpropagation pass (end of batch)
	virtual void update(float* params, const float* data, size_t batchSize) = 0;
};
}
}
