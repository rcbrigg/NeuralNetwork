inline float sigmoid(half x)
{
    return 1.f / (1.f + exp_half(-x));
}

inline float sigmoidPrime(float x)
{
    const half s = sigmoid(x);
    return s * (1.f - s);
}

__kernel void forward(__global const half* input,
                      __global half* output)
{
    const gid = get_global_id(0);
    output[gid] = sigmoid(input[gid]);
}

__kernel void backPropagate(__global const half* input,
                            __global const half* outputError,
                            __global half* inputError)
{
    const gid = get_global_id(0);
    output[gid] = sigmoidPrime(input[gid]) * outputError[gid];
}