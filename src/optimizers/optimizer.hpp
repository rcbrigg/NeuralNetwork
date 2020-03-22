#pragma once
#include "tensor.hpp"

namespace nn
{
namespace optimizer
{
class Optimizer
{
public:
	virtual Tensor<> getDerivatives() = 0;

	virtual void update(float* params, size_t batchSize) = 0;
};
}
}
