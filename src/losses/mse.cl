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
	uint stride = get_global_size(0);

	for (uint i = gid; i < size; i += stride)
	{
		error[i] = square(output[i] - target[i]);
	}
}

__kernel void calculateTotalError(__global const float* output,
								  __global const float* target,
								  __global float* error,
								  uint width,
								  uint size)
{
	const size_t gid = get_global_id(0);
	const size_t wid = get_group_id(0);
	const size_t lid = get_local_id(0);
	const size_t stride = get_local_size(0);
	__local float temp[WORKGROUP_SIZE];
	temp[lid] = 0;

	const __global float* outSet = output + width * wid;
	const __global float* trgSet = target + width * wid;
	for (uint i = lid; i < width; i += stride)
	{
		temp[lid] += square(outSet[i] - trgSet[i]);
	}

	for (uint i = stride / 2; i > 0; i /= 2)
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
	uint stride = get_global_size(0);
	for (uint i = gid; i < size; i += stride)
	{
		derivatives[i] = 2 * (output[i] - target[i]);
	}
}