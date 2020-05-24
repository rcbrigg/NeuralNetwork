__kernel void update(__global float* params, __global float* deltas, float scale, uint size)
{
    const size_t gid = get_global_id(0);
	const size_t stride = get_global_size(0);
	for (uint i = gid; i < size; i += stride)
	{
		params[i] -= scale * deltas[i];
		deltas[i] = 0;
	}
}

// TODO: replace with clear
__kernel void init(__global float* params, uint paramCount)
{
	const size_t gid = get_global_id(0);
	if (gid < paramCount)
		params[gid] = 0;
}