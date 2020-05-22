inline float sigmoid(float x)
{
	return 1.f / (1.f + exp(-x));
}

inline float sigmoidPrime(float x)
{
	float s = sigmoid(x);
	return s * (1.f - s);
}

__kernel void forward(__global const float* input,
					  __global float* output,
					  const uint inputOffset,
					  const uint outputOffset)
{
	input += inputOffset;
	output += outputOffset;

	const size_t gid = get_global_id(0);
	output[gid] = sigmoid(input[gid]);
}

__kernel void backPropagate(__global const float* input,
							__global const float* outputError,
							__global float* inputError,
							const uint inputOffset)
{
	input += inputOffset;
	const size_t gid = get_global_id(0);
	inputError[gid] = sigmoidPrime(input[gid]) * outputError[gid];
}