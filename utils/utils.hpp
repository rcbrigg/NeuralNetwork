#pragma once
#include "..\include\tensor.hpp"
#include <random>

namespace nn
{

inline float fastUniformRand(float min, float max)
{
	static uint32_t seed = 0;
	seed = 214013 * seed + 2531011;
	return (float((seed >> 16) & 0x7FFF) / float(0x7FFF)) * (max - min) + min;
}

inline Tensor<> uniformRandomTensor(size_t size, float min, float max)
{
	static std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(min, max);

	auto tensor = Tensor<>(size);

	for (size_t i = 0; i < size; ++i)
	{
		tensor[i] = distribution(generator);
	}

	return tensor;
}

template<typename T> size_t argMax(const T* args, size_t count)
{
	size_t argMax = 0;
	for (size_t i = 1; i < count; ++i)
	{
		if (args[i] > args[argMax])
		{
			argMax = i;
		}
	}
	return argMax;
}

inline bool AreWithinTolerance(const float* a, const float* b, size_t size, float delta)
{
	for (size_t i = 0; i < size; ++i)
	{
		if (abs(a[i] - b[i]) > delta)
		{
			return false;
		}
	}
	return true;
}
}