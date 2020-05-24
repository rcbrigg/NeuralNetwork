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
						 const uint inputWidth)
{
	const float sd = rsqrt((double)inputWidth);
	const size_t stride = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	const size_t outputSize = get_num_groups(0);
	const size_t offset = outputSize + wid * inputWidth;
	__global float* weights = params + offset;
	uint seed = get_global_id(0);

	for (size_t i = lid ; i < inputWidth; i += stride)
	{
		weights[i] = fastRand(-sd, sd, &seed);
	}

	if (lid == 0)
	{
		__global float* bias = params + wid;
		bias[wid] = 0;
	}
}

__kernel void forward(__global const float* input,
					  __global float* output,
					  __global const float* params,
					  const uint inputOffset,
					  const uint outputOffset,
					  const uint inputWidth,
					  const uint outputWidth,
					  const uint outputSize)
{
	input += inputOffset;
	output += outputOffset;

	__local float temp[WORKGROUP_SIZE];

	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	const size_t groupCount = get_num_groups(0);

	// Todo: optimise group working on multiple rows if rows are small
	for (uint j = wid; j < outputSize; j += groupCount)
	{
		uint row = wid;// j% outputWidth;
		const __global float* weights = params + outputWidth + row * inputWidth; // row of weight matrix

		temp[lid] = 0.f;

		// sum
		for (uint i = lid; i < inputWidth; i += localSize)
		{
			temp[lid] += weights[i] * input[i];
		}

		// accumulate
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
			output[j] = params[row] + temp[0];
		}
	}
}

__kernel void backPropagate(__global const float* outputError,
							__global float* inputError,
							__global const float* params,
							const uint ACTUAL_GID, // TODO REMOVE
							const uint inputWidth,
							const uint outputWidth)
{
	__local float temp[WORKGROUP_SIZE];

	size_t gid = get_global_id(0);

	if (gid < ACTUAL_GID)
	{
		const uint col = gid % inputWidth;
		outputError += gid / inputWidth;

		const __global float* weights = params + outputWidth;
		float sum = 0;

		for (size_t i = col, j = 0; j < outputWidth; i += inputWidth, j++)
		{
			sum += weights[i] * outputError[j];
		}

		inputError[gid] = sum;
	}
}

__kernel void calculateDerivatives(__global const float* input,
								   __global const float* outputError,
								   __global float* derivatives,
								   const uint inputOffset,
								   const uint inputWidth,
								   const uint outputWidth,
								   const uint outputSize)
{
	input += inputOffset;
	const size_t groupSize = get_local_size(0);
	const size_t wid = get_group_id(0);
	const size_t lid = get_local_id(0);
	const size_t stride = get_num_groups(0);

	// Todo: optimise group working on multiple rows if rows are small
	uint row = wid;//% outputWidth;
	__global float* dw = derivatives + outputWidth + row * inputWidth;

	for (uint j = wid; j < outputSize; j += stride)
	{	
		for (size_t i = lid; i < inputWidth; i += groupSize)
		{
			dw[i] += outputError[j] * input[i];
		}

		if (lid == 0)
		{
			derivatives[row] += outputError[j];
		}
	}
}