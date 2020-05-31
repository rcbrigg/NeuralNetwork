#define beta1 (0.9f)
#define beta2 (0.999f)
#define epsilon (1e-8)

__kernel void update(__global float* params, __global float* deltas, __global float* m, __global float* v, __global const float* betaPow, float scale, uint size)
{
	const size_t gid = get_global_id(0);
	const size_t stride = get_global_size(0);
	for (uint i = gid; i < size; i += stride)
	{
		float g = deltas[i];
		m[i] = beta1 * m[i] + (1 - beta1) * g;
		v[i] = beta2 * v[i] + (1 - beta2) * g * g;
		float mHat = m[i] / (1 - betaPow[0]);
		float vHat = v[i] / (1 - betaPow[1]);
		params[i] -= scale * mHat / (sqrt(vHat) + epsilon);
		deltas[i] = 0;
	}
}

__kernel void updateBetas(__global float* betaPow)
{
	betaPow[0] *= beta1;
	betaPow[1] *= beta2;
}

