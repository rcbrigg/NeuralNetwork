#define WORKGROUP_SIZE (32)

inline uint updateSeed(uint seed)
{
	return seed = 0xa6718293 * seed + 0x638c571f;
}

inline float fastRand(float min, float max, uint* seed)
{
	*seed = updateSeed(*seed);
	return ((float)((*seed >> 16) & 0x7FFF) / (float)(0x7FFF)) * (max - min) + min;
}

__kernel void initParams(__global float* params,
						 const uint inputSize)
{
	const float sd = rsqrt((double)inputSize);
	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	const size_t outputSize = get_num_groups(0);
	const size_t offset = outputSize + wid * inputSize;
	__global float* weights = params + offset;
	uint seed = get_global_id(0);

	for (size_t i = lid ; i < inputSize; i += localSize)
	{
		weights[i] = fastRand(-sd, sd, &seed);
	}

	if (lid == 0)
	{
		__global float* bias = params + wid;
		bias[wid] = 0;
	}
}

// Each workgroup computes a single output element
__kernel void forward(__global const float* input,
					  __global float* output,
					  __global const float* params,
					  const uint inputOffset,
					  const uint outputOffset,
					  const uint inputSize)
{
	input += inputOffset;
	output += outputOffset;

	__local float temp[WORKGROUP_SIZE];

	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	const size_t outputSize = get_num_groups(0);
	const size_t offset = outputSize + wid * inputSize;
	const __global float* row = params + offset; // row of weight matrix

	temp[lid] = 0.f;

	for (uint i = lid; i < inputSize; i += localSize)
	{
		temp[lid] += row[i] * input[i];
	}

	for (uint i = localSize / 2; i > 0; i /= 2)
	{
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < i)
		{
			temp[lid] += temp[lid + i];
		}
	}

	if (lid == 0)
	{
		// bias + weight matrix row product
		output[wid] = params[wid] + temp[0];
	}
}

__kernel void backPropagate(__global const float* outputError,
							__global float* inputError,
							__global const float* params,
							const uint inputSize,
							const uint outputSize)
{
	__local float temp[WORKGROUP_SIZE];

	size_t gid = get_global_id(0);

	if (gid < inputSize)
	{
		size_t lid = get_local_id(0);
		const __global float* weights = params + outputSize;
		temp[lid] = 0;

		for (size_t i = gid, j = 0; j < outputSize; i += inputSize, j++)
		{
			temp[lid] += weights[i] * outputError[j];
		}

		inputError[gid] = temp[lid];
	}
	
}

__kernel void calculateDerivatives(__global const float* input,
								   __global const float* outputError,
								   __global float* derivatives,
								   const uint inputOffset,
								   const uint inputSize)
{
	input += inputOffset;
	const size_t localSize = get_local_size(0);
	const size_t outputSize = get_num_groups(0);
	const size_t wid = get_group_id(0);
	const size_t lid = get_local_id(0);
	__global float* dw = derivatives + outputSize + wid * inputSize;

	for (size_t i = lid; i < inputSize; i += localSize)
	{
		dw[i] += outputError[wid] * input[i];
	}

	if (lid == 0)
	{
		derivatives[wid] += outputError[wid];
	}
}