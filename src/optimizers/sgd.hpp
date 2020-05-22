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
		initBatchKernel = clCreateKernel(program, "initBatch", &error);
		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for optimizer::sgd.");
		}

		parameterCount = paramCount;
	}

	void cl_beginBatch(cl_command_queue queue, cl_mem derivatives)
	{
		int error;
		error = clSetKernelArg(initBatchKernel, 0, sizeof(cl_mem), &derivatives);
		error |= clEnqueueNDRangeKernel(queue, initBatchKernel, 1, NULL, &parameterCount, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::sgd::cl_beginBatch()");
		}
	}

	void cl_update(cl_command_queue queue, cl_mem parameters, cl_mem derivatives, size_t batchSize)
	{
		const float scale = learningRate / batchSize;

		int error;
		error = clSetKernelArg(updateKernel, 0, sizeof(cl_mem), &parameters);
		error |= clSetKernelArg(updateKernel, 1, sizeof(cl_mem), &derivatives);
		error |= clSetKernelArg(updateKernel, 2, sizeof(scale), &scale);
		error |= clEnqueueNDRangeKernel(queue, updateKernel, 1, NULL, &parameterCount, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::sgd::cl_update()");
		}
	}

private:
	float learningRate;

	size_t parameterCount;

	cl_kernel updateKernel;

	cl_kernel initBatchKernel;
};
}
}
