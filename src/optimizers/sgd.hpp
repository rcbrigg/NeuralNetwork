#pragma once
#include "optimizer.hpp"

namespace nn
{
namespace optimizer
{
class Sgd : public Optimizer
{
public:
	Sgd(float learningRate) :
		learningRate(learningRate),
		parameterCount(0),
		updateKernel(NULL)
	{
	}

	void init(float* data, size_t paramCount) final
	{
		parameterCount = paramCount;
	}

	void beginBatch(float* data) final
	{
		memset(data, 0, sizeof(float) * parameterCount);
	}

	void update(float* params, const float* data, size_t batchSize) final
	{
		const size_t size = parameterCount;
		const float scale = learningRate / batchSize;
		const float* derivative = data;

		for (size_t i = 0; i < parameterCount; ++i)
		{
			params[i] -= scale * derivative[i];
		}
	}

	void cl_init(cl_context context, cl_device_id device, cl_command_queue, cl_mem, size_t paramCount) final
	{
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		updateKernel = clCreateKernel(program, "update", &error);
		initKernel = clCreateKernel(program, "init", &error);
		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for optimizer::sgd.");
		}

		parameterCount = paramCount;
	}

	void cl_beginTraining(cl_command_queue queue, cl_mem derivatives) final
	{
		int error;
		uint32_t size = parameterCount;
		size_t globalSize = cl::alignSize(size);
		error = clSetKernelArg(initKernel, 0, sizeof(cl_mem), &derivatives);
		error = clSetKernelArg(initKernel, 1, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, initKernel, 1, NULL, &globalSize, &cl::workGroupSize, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::sgd::cl_beginBatch()");
		}
	}

	void cl_update(cl_command_queue queue, cl_mem parameters, cl_mem derivatives, size_t batchSize)
	{
		const float scale = learningRate / batchSize;
		uint32_t size = parameterCount;
		size_t globalSize = cl::alignSize(size);
		int error;
		
		error = clSetKernelArg(updateKernel, 0, sizeof(cl_mem), &parameters);
		error |= clSetKernelArg(updateKernel, 1, sizeof(cl_mem), &derivatives);
		error |= clSetKernelArg(updateKernel, 2, sizeof(scale), &scale);
		error |= clSetKernelArg(updateKernel, 3, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, updateKernel, 1, NULL, &globalSize, &cl::workGroupSize, 0, NULL, NULL);

		
		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::sgd::cl_update()");
		}
	}

private:
	float learningRate;

	size_t parameterCount;

	cl_kernel updateKernel;

	cl_kernel initKernel;
};
}
}
