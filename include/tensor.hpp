#pragma once
#include "shape.hpp"

namespace nn
{


template<size_t N = 1, typename Type = float>
class Tensor
{
public:
	Tensor(size_t size, Type* data);

	using value_type = Type;

	size_t size();

	size_t length(size_t dimension = 0);

	value_type* data();

private:
};

template<size_t N = 1, typename Type = float>
using ConstTensor = Tensor<N, const Type>;

}
