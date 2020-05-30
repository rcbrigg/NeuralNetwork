#define MAX_WORKGROUP_SIZE (256)

//__kernel void uploadInputDataDefault(__global const float* src, __global float* dst, const uint size, const uint stride, const uint srcSize)
//{
//	const size_t gid = get_global_id(0);
//	const size_t i = gid / size;
//	const size_t j = gid % size;
//	if (gid < srcSize)
//		dst[stride * i + j] = src[gid];
//}
//
//__kernel void downloadOutputData(__global const float* src, __global float* dst, const uint size, const uint stride, const uint dstSize)
//{
//	const size_t gid = get_global_id(0);
//	const size_t i = gid / size;
//	const size_t j = gid % size;
//	if (gid < dstSize)
//		dst[gid] = src[stride * i + j];
//}

__kernel void softmaxError(__global const float* output,
						   __global const uint* target,
						   __global float* outputError,
						   uint targetOffset,
						   uint outputWidth,
						   uint size)
{
	target += targetOffset;
	const uint gid = get_global_id(0);
	const uint stride = get_global_size(0);

	for (uint i = gid; i < size; i += stride)
	{
		outputError[i] = output[i];
		if ((i % outputWidth) == target[i / outputWidth])
		{
			outputError[gid] -= 1.0f;
		}
	}
}

__kernel void classify(__global const float* src,
					   __global uint* output,
					   const uint stride,
					   const uint size)
{
	__local float maxValue[MAX_WORKGROUP_SIZE];
	__local uint maxPos[MAX_WORKGROUP_SIZE];
	const size_t localSize = get_local_size(0);
	const size_t lid = get_local_id(0);
	const size_t wid = get_group_id(0);
	__global const float* srcSet = src + wid * stride;

	// Find position of max element
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
		barrier(CLK_LOCAL_MEM_FENCE);
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