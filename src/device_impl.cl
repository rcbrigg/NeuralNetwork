#define WORKGROUP_SIZE (32)

__kernel void uploadInputDataDefault(__global const float* src, __global float* dst, const uint size, const uint stride, const uint srcSize)
{
	const size_t gid = get_global_id(0);
	const size_t i = gid / size;
	const size_t j = gid % size;
	if (gid < srcSize)
		dst[stride * i + j] = src[gid];
}

__kernel void downloadOutputData(__global const float* src, __global float* dst, const uint size, const uint stride, const uint dstSize)
{
	const size_t gid = get_global_id(0);
	const size_t i = gid / size;
	const size_t j = gid % size;
	if (gid < dstSize)
		dst[gid] = src[stride * i + j];
}

__kernel void classifierCalcDerivatives(__global const float* output,
										__global const uint* target,
										__global float* outputError,
										uint targetOffset,
										int size)
{
	target += targetOffset;
	const size_t gid = get_global_id(0);

	if (gid == *target)
	{
		outputError[gid] = output[gid] - 1.0f;
	}
	else if (gid < size)
	{
		outputError[gid] = output[gid];
	}
}

__kernel void classifyOutputDataDefault(__global const float* src, __global uint* output, const uint stride, const uint size)
{
	__local float maxValue[WORKGROUP_SIZE];
	__local float maxPos[WORKGROUP_SIZE];
	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	__global const float* srcSet = src + wid * stride;

	// Find position o max element
	if (lid < size)
	{
		maxValue[lid] = srcSet[lid];
		maxPos[lid] = lid;
	}
	else
	{
		maxValue[lid] = -MAXFLOAT;
	}	

	for (size_t i = lid + localSize; i < size; i += localSize)
	{
		if (maxValue[lid] < srcSet[i])
		{
			maxValue[lid] = srcSet[lid];
			maxPos[lid] = i;
		}
	}

	for (size_t i = localSize / 2; i > 0; i /= 2)
	{
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < i)
		{
			if (maxValue[lid] < maxValue[lid + i])
			{
				maxValue[lid] = maxValue[lid + i];
				maxPos[lid] = maxPos[lid + i];
			}
		}
	}

	if (lid == 0)
	{
		output[wid] = maxPos[0];
	}
}