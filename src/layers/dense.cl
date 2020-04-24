#define MAX_WORKGROUP_SIZE (256)

// Each workgroup computes a single output element
__kernel void forward(__global const half* input,
					  __global half* output,
					  __global const half* params
					  const size_t inputSize)
{
	__local half temp[MAX_WORKGROUP_SIZE];
	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const __global half* weight = params;
	const size_t step = (inputSize + localSize - 1) / localSize;

	temp[lid] = weight[lid] * input[lid];
	for (size_t i = lid + step; i < inputSize; i += step)
	{
		temp[lid] += weight[i] * input[i];
	}

	for (size_t i = localSize / 2; i > 0; i /= 2)
	{
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (lid > i)
		{
			temp[lid] += temp[lid + i];
		}
	}

	if (lid == 0)
	{
		const wid = get_workgroup_id(1);
		const __global half* bias = params + wid * outputSize * inputSize;
		output[wid] = bias[0] + temp[0];
	}
}

__kernel void backPropagate(__global const half* input,
							__global const half* outputError
							__global half* inputError,
							__global const half* params
							const size_t outputSize)
{
	__local half temp[MAX_WORKGROUP_SIZE];
	const size_t localSizeX = get_local_size(0);
	const size_t localSizeY = get_local_size(1);
	const size_t widX = get_workgroup_id(0);
	const size_t widY = get_workgroup_id(1);
	const size_t lidX = get_local_id(0);
	const size_t lidY = get_local_id(1);
	const __global half* weight = params;

	if (localSizeX == 0)
	{
		temp[lidY] = outputError[localSizeY * widY + lidY];
	}

	temp[lid] = weight[lid] * input[lid];
	for (size_t i = lid + step; i < inputSize; i += step)
	{
		[lid] += weight[i] * input[i];
	}

	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	// Now sum all elements in temp
	for (size_t i = localSize / 2; i > 0; i /= 2)
	{
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (lid > i)
		{
			temp[lid] += temp[lid + i];
		}
	}

	if (lid == 0)
	{
		output[get_workgroup_id(1)] = bias[0] + temp[0];
	}
}

__kernel void calculateDerivatives(__global const half* input,
							__global const half* outputError
							__global half* derivatives,
								   const size_t inputSize
							const size_t inpuSize)
{
	const size_t localSize = get_local_size(0);
	const size_t wid = get_workgroup_id(1);
	const size_t lid = get_local_id(0);
	__global half* dw = derivatives;
	const size_t step = (inputSize + localSize - 1) / localSize;

	temp[lid] = weight[lid] * input[lid];
	for (size_t i = lid + step; i < inputSize; i += step)
	{
		dw[i] += outputError[wid] * input[i];
	}

	if (lid == 0)
	{	
		__global half* db = derivatives + wid * outputSize * inputSize;
		db[wid] += outputError[wid];
	}
}