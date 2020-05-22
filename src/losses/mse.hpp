#pragma once
#include "loss.hpp"

namespace nn
{
namespace loss
{
class Mse : public Loss
{
public:

	void calculateError(const float* output, const float* target, float* error, size_t size) const final
	{
		for (size_t i = 0; i < size; ++i)
		{
			error[i] = square(output[i] - target[i]);
		}
	}

	float calculateError(const float* output, const float* target, size_t size) const final
	{
		float error = 0;
		for (size_t i = 0; i < size; ++i)
		{
			error += square(output[i] - target[i]);
		}
		return error;
	}

	void calculateDerivatives(const float* output, const float* target, float* derivatives, size_t size) const final
	{
		for (size_t i = 0; i < size; ++i)
		{
			derivatives[i] = 2.f * (output[i] - target[i]);
		}
	}

	void cl_calculateError(cl_command_queue queue, cl_mem output, cl_mem target, cl_mem ouputError, uint32_t targetOffset, uint32_t size) const  final
	{
		size_t globalSize = cl::alignSize(size);

		int error;
		error = clSetKernelArg(calculateErrorKernel, 0, sizeof(cl_mem), &output);
		error |= clSetKernelArg(calculateErrorKernel, 1, sizeof(cl_mem), &target);
		error |= clSetKernelArg(calculateErrorKernel, 2, sizeof(cl_mem), &ouputError);
		error |= clSetKernelArg(calculateErrorKernel, 3, sizeof(targetOffset), &targetOffset);
		error |= clSetKernelArg(calculateErrorKernel, 4, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, calculateErrorKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_calculateDerivatives()");
		}
	}

	void cl_calculateTotalError(cl_command_queue queue, cl_mem output, cl_mem target, cl_mem ouputError, uint32_t size, size_t count) const  final
	{
		size_t globalSize = count * cl::DEFAULT_WORKGROUP_SIZE;
		uint32_t stride = cl::alignSize(size);

		int error;
		error = clSetKernelArg(calculateTotalErrorKernel, 0, sizeof(cl_mem), &output);
		error |= clSetKernelArg(calculateTotalErrorKernel, 1, sizeof(cl_mem), &target);
		error |= clSetKernelArg(calculateTotalErrorKernel, 2, sizeof(cl_mem), &ouputError);
		error |= clSetKernelArg(calculateTotalErrorKernel, 3, sizeof(stride), &stride);
		error |= clSetKernelArg(calculateTotalErrorKernel, 4, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, calculateTotalErrorKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_calculateDerivatives()");
		}
	}

	void cl_calculateDerivatives(cl_command_queue queue, cl_mem output, cl_mem target, cl_mem derivatives, uint32_t targetOffset, uint32_t size) const final
	{
		size_t globalSize = cl::alignSize(size);

		int error;
		error = clSetKernelArg(calculateDerivativesKernel, 0, sizeof(cl_mem), &output);
		error |= clSetKernelArg(calculateDerivativesKernel, 1, sizeof(cl_mem), &target);
		error |= clSetKernelArg(calculateDerivativesKernel, 2, sizeof(cl_mem), &derivatives);
		error |= clSetKernelArg(calculateDerivativesKernel, 3, sizeof(targetOffset), &targetOffset);
		error |= clSetKernelArg(calculateDerivativesKernel, 4, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, calculateDerivativesKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_calculateDerivatives()");
		}
	}

	void cl_initKernels(cl_context context, cl_device_id device)
	{
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		calculateErrorKernel = clCreateKernel(program, "calculateError", &error);
		calculateDerivativesKernel = clCreateKernel(program, "calculateDerivatives", &error);
		calculateTotalErrorKernel = clCreateKernel(program, "calculateTotalError", &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for mse::cl_initKernels().");
		}
	}

private:
	static float square(float x) { return x * x; }

	cl_kernel calculateErrorKernel;

	cl_kernel calculateDerivativesKernel;

	cl_kernel calculateTotalErrorKernel;
};
}
}
