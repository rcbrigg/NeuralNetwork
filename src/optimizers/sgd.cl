__kernel void update(__global float* params, __global const float* deltas, float scale)
{
    const size_t gid = get_global_id(0);
	params[gid] -= scale * deltas[gid];
}

__kernel void initBatch(__global float* params)
{
	const size_t gid = get_global_id(0);
	params[gid] = 0;
}