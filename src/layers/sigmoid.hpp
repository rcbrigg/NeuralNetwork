#pragma once
#include "layer.hpp"
#include "../cl/cl_utils.hpp"
#include <math.h>
#include <cstring>

namespace nn
{
namespace layer
{
class Sigmoid : public Layer
{
public:
	Sigmoid(size_t size) :
		Layer(size, size, 0)
	{
	}

	void forward(const float* input, const float*, float* output) const final
	{
		for (size_t i = 0; i < inputSize; ++i)
		{
			output[i] = sigmoid(input[i]);
		}
	}

	void backPropagate(const BackPropData& data, float* inputError) const final
	{
		for (size_t i = 0; i < inputSize; ++i)
		{
			inputError[i] = sigmoidPrime(data.input[i]) * data.outputError[i];
		}
	}

	void cl_forward(cl_command_queue queue, cl_mem input, cl_mem, cl_mem output, uint32_t inOffset, uint32_t outOffset, uint32_t batchSize) const final
	{
		uint32_t size = batchSize * inputSize;
		size_t globalSize = cl::alignSize(size);

		int error;
		error = clSetKernelArg(forwardKernel, 0, sizeof(cl_mem), &input);
		error |= clSetKernelArg(forwardKernel, 1, sizeof(cl_mem), &output);
		error |= clSetKernelArg(forwardKernel, 2, sizeof(inOffset), &inOffset);
		error |= clSetKernelArg(forwardKernel, 3, sizeof(outOffset), &outOffset);
		error |= clSetKernelArg(forwardKernel, 4, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, forwardKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::sigmoid::cl_forward()");
		}
	}

	void cl_backPropagate(cl_command_queue queue, const ClBackPropData& data, cl_mem inputError, uint32_t batchSize) const final
	{
		uint32_t size = batchSize * inputSize;
		size_t globalSize = cl::alignSize(size);

		int error;
		error = clSetKernelArg(backPropagateKernel, 0, sizeof(cl_mem), &data.input);
		error |= clSetKernelArg(backPropagateKernel, 1, sizeof(cl_mem), & data.outputError);
		error |= clSetKernelArg(backPropagateKernel, 2, sizeof(cl_mem), &inputError);
		error |= clSetKernelArg(backPropagateKernel, 3, sizeof(data.inputOffset), &data.inputOffset);
		error |= clSetKernelArg(backPropagateKernel, 4, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, backPropagateKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::sigmoid::cl_backPropagate()");
		}
	}

	void cl_initKernels(cl_context context, cl_device_id device) final
	{
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		forwardKernel = clCreateKernel(program, "forward", &error);
		backPropagateKernel = clCreateKernel(program, "backPropagate", &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for layer::cl_initKernels().");
		}
	}

	static float sigmoid(float x)
	{
		return 1.f / (1.f + expf(-x));
	}

	static float sigmoidPrime(float x)
	{
		const float s = sigmoid(x);
		return s * (1.f - s);
	}
private:
	cl_kernel forwardKernel;

	cl_kernel backPropagateKernel;

};
}
}