#pragma once
#include "..\..\include\shape.hpp"
namespace nn
{
namespace layer
{
class Layer
{
public:
	virtual void forward(const float* in, float* out) = 0;
	const Shape& getInputShape() const { return inputShape; }
	const Shape& getOutputShape() const { return outputShape; }

protected:
	Layer(Shape inputShape, Shape outputShape);

	const Shape inputShape;
	const Shape outputShape;
};
}
}