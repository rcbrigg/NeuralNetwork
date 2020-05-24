inline float sigmoid(float x)
{
	return 1.f / (1.f + exp(-x));
}

inline float sigmoidPrime(float x)
{
	float s = sigmoid(x);
	return s * (1.f - s);
}

__kernel void forward(__global const float* input, // layer input vector
					  __global float* output,      // layer output vector
					  const uint inputOffset,      // offset into input
					  const uint outputOffset,     // offset into output
					  const uint size)             // length of input (and output) vector
{
	input += inputOffset;
	output += outputOffset;
	uint gid = get_global_id(0);
	uint stride = get_global_size(0);

	for (uint i = gid; i < size; i += stride)
	{		
		output[i] = sigmoid(input[i]);
	}
}

__kernel void backPropagate(__global const float* input,       // layer input vector
							__global const float* outputError, // layer output error vector
							__global float* inputError,        // OUTPUT -> layer input error vector
							const uint inputOffset,            // offset into innput
							const uint size)				   // length of input vector
{
	input += inputOffset;
	uint gid = get_global_id(0);
	uint stride = get_global_size(0);

	for (uint i = gid; i < size; i += stride)
	{
		inputError[i] = sigmoidPrime(input[i]) * outputError[i];
	}
	
}