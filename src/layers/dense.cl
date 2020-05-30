#define MAX_WORKGROUP_SIZE (256)

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
						 const uint inputWidth,
						 const uint paramOffset)
{
	params += paramOffset;
	const float sd = rsqrt((double)inputWidth);
	const size_t stride = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	const size_t outputSize = get_num_groups(0);
	const size_t offset = outputSize + wid * inputWidth;
	__global float* weights = params + offset;
	uint seed = get_global_id(0);

	for (size_t i = lid; i < inputWidth; i += stride)
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
					  const uint paramOffset,
					  const uint inputWidth,
					  const uint outputWidth,
					  const uint outputSize)
{
	input += inputOffset;
	output += outputOffset;
	params += paramOffset;

	__local float temp[MAX_WORKGROUP_SIZE];

	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	//const size_t groupCount = get_num_groups(0);

	// Todo: optimise group working on multiple rows if rows are small
	//for (uint j = wid; j < outputSize; j += groupCount)
	if (wid < outputSize)
	{
		uint set = wid / outputWidth;
		uint row = wid - set * outputWidth;;
		const __global float* weights = params + outputWidth + row * inputWidth; // row of weight matrix
		input += set * inputWidth;
		temp[lid] = 0.f;

		// sum
		for (uint i = lid; i < inputWidth; i += localSize)
		{
			temp[lid] += weights[i] * input[i];
		}

		// accumulate
		for (uint i = localSize / 2; i > 0; i /= 2)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if (lid < i)
			{
				temp[lid] += temp[lid + i];
			}
		}

		if (lid == 0)
		{
			// bias + weight matrix row product
			output[wid] = params[row] + temp[0];
		}
	}
}

__kernel void backPropagate(__global const float* outputError,
							__global float* inputError,
							__global const float* params,
							const uint paramOffset,
							const uint inputWidth,
							const uint outputWidth)
{
	params += paramOffset;

	__local float temp[MAX_WORKGROUP_SIZE];

	size_t col = get_global_id(0);
	size_t row = get_global_id(1);

	if (col < inputWidth)
	{
		const __global float* weights = params + outputWidth;
		const __global float* outputError_ = outputError + row * outputWidth;
		__global float* inputError_ = inputError + row * inputWidth;
		float sum = 0;

		for (size_t i = col, j = 0; j < outputWidth; i += inputWidth, j++)
		{
			sum += weights[i] * outputError_[j];
		}

		inputError_[col] = sum;
	}
}

__kernel void calculateDerivatives(__global const float* input,
								   __global const float* outputError,
								   __global float* derivatives,
								   const uint inputOffset,
								   const uint paramOffset,
								   const uint inputWidth,
								   const uint outputWidth,
								   const uint outputSize)
{
	derivatives += paramOffset;

	input += inputOffset;
	size_t col = get_global_id(0);
	size_t row = get_global_id(1);

	if (col < inputWidth)
	{
		__global float* dw = derivatives + outputWidth + row * inputWidth + col;

		for (uint i = col, j = row; j < outputSize; j += outputWidth, i += inputWidth)
		{
			*dw += outputError[j] * input[i];
			if (col == 0)
			{
				derivatives[row] += outputError[j];
			}
		}
	}
}