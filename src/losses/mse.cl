#define WORKGROUP_SIZE (32)

static float square(float x) { return x * x; }

__kernel void calculateError(__global const float* output,
							 __global const float* target,
							 __global float* error,
							 uint targetOffset,
							 uint size)
{
	target += targetOffset;

	const size_t gid = get_global_id(0);
	if (gid < size)
	{
		error[gid] = square(output[gid] - target[gid]);
	}
}

__kernel void calculateTotalError(__global const float* output,
								  __global const float* target,
								  __global float* error,
								  uint stride,
								  uint size)
{
	const size_t gid = get_global_id(0);
	const size_t wid = get_group_id(0);
	const size_t lid = get_local_id(0);
	const size_t localSize = get_local_size(0);
	__local float temp[WORKGROUP_SIZE];
	temp[lid] = 0;

	const __global float* outSet = output + stride * wid;
	const __global float* trgSet = target + stride * wid;
	for (uint i = lid; i < size; i += localSize)
	{
		temp[lid] += square(outSet[i] - trgSet[i]);
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
		error[wid] = temp[0];
	}
}

__kernel void calculateDerivatives(__global const float* output,
								   __global const float* target,
								   __global float* derivatives,
								   uint targetOffset,
								   uint size)
{
	target += targetOffset;

	const size_t gid = get_global_id(0);
	if (gid < size)
	{
		derivatives[gid] = 2 * (output[gid] - target[gid]);
	}
}