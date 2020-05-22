#pragma once
#include "layer.hpp"
#include "..\..\utils\utils.hpp"

namespace nn
{
namespace layer
{
class Dense : public Layer
{
public:
	Dense(size_t inputSize, size_t outputSize) :
		Layer(inputSize, outputSize, (inputSize + 1) * outputSize),
		forwardKernel(NULL),
		backPropagateKernel(NULL),
		calculateDerivativesKernel(NULL)
	{
	}

	void forward(const float* input, const float* params, float* output) const final
	{
		const float* bias = getBiases(params);
		const float* weight = getWeights(params);
		
		for (size_t i = 0; i < outputSize; ++i)
		{
			output[i] = *bias++;
			for (size_t j = 0; j < inputSize; ++j)
			{
				output[i] += input[j] * *weight++;
			}
		}
	}

	void backPropagate(const BackPropData& data, float* inputError) const final
	{
		memset(inputError, 0, inputSize * sizeof(float));

		const float* weight = getWeights(data.params);

		for (size_t i = 0; i < outputSize; ++i)
		{
			for (size_t j = 0; j < inputSize; ++j)
			{
				inputError[j] += data.outputError[i] * *weight++;
			}
		}
	}

	void calculateDerivatives(const BackPropData& data, float* derivatives) const final
	{
		float* db = getBiases(derivatives);
		float* dw = getWeights(derivatives);

		for (size_t i = 0; i < outputSize; ++i)
		{
			*db++ += data.outputError[i];
			for (size_t j = 0; j < inputSize; ++j)
			{
				*dw++ += data.outputError[i] * data.input[j];
			}
		}
	}

	void initializeParameters(float* params) const final
	{
		const auto sd = 1.f / (float)sqrt(getInputSize());
		float* bias = getBiases(params);
		float* weight = getWeights(params);

		for (size_t i = 0; i < outputSize; ++i)
		{
			*bias++ = fastUniformRand(-1.f, 1.f);
			for (size_t j = 0; j < inputSize; ++j)
			{
				*weight++ = fastUniformRand(-sd, sd);
			}
		}
	}

	void cl_forward(cl_command_queue queue, cl_mem input, cl_mem params, cl_mem output, uint32_t inOffset, uint32_t outOffset) const final
	{
		size_t globalSize = cl::DEFAULT_WORKGROUP_SIZE * outputSize;

		int error;
		error = clSetKernelArg(forwardKernel, 0, sizeof(cl_mem), &input);
		error |= clSetKernelArg(forwardKernel, 1, sizeof(cl_mem), &output);
		error |= clSetKernelArg(forwardKernel, 2, sizeof(cl_mem), &params);
		error |= clSetKernelArg(forwardKernel, 3, sizeof(inOffset), &inOffset);
		error |= clSetKernelArg(forwardKernel, 4, sizeof(outOffset), &outOffset);
		error |= clSetKernelArg(forwardKernel, 5, sizeof(inputSize), &inputSize);
		error |= clEnqueueNDRangeKernel(queue, forwardKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_forward()");
		}
	}

	void cl_backPropagate(cl_command_queue queue, const ClBackPropData& data, cl_mem inputError) const final
	{
		size_t globalSize = inputStride;

		int error;
		error = clSetKernelArg(backPropagateKernel, 0, sizeof(cl_mem), &data.outputError);
		error |= clSetKernelArg(backPropagateKernel, 1, sizeof(cl_mem), &inputError);
		error |= clSetKernelArg(backPropagateKernel, 2, sizeof(cl_mem), &data.params);
		error |= clSetKernelArg(backPropagateKernel, 3, sizeof(inputSize), &inputSize);
		error |= clSetKernelArg(backPropagateKernel, 4, sizeof(outputSize), &outputSize);
		error |= clEnqueueNDRangeKernel(queue, backPropagateKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_calculateDerivatives()");
		}
	}

	void cl_calculateDerivatives(cl_command_queue queue, const ClBackPropData& data, cl_mem derivaitves) const final
	{
		size_t globalSize = cl::DEFAULT_WORKGROUP_SIZE * outputSize;

		int error;
		error = clSetKernelArg(calculateDerivativesKernel, 0, sizeof(cl_mem), &data.input);
		error |= clSetKernelArg(calculateDerivativesKernel, 1, sizeof(cl_mem), &data.outputError);
		error |= clSetKernelArg(calculateDerivativesKernel, 2, sizeof(cl_mem), &derivaitves);
		error |= clSetKernelArg(calculateDerivativesKernel, 3, sizeof(data.inputOffset), &data.inputOffset);
		error |= clSetKernelArg(calculateDerivativesKernel, 4, sizeof(inputSize), &inputSize);
		error |= clEnqueueNDRangeKernel(queue, calculateDerivativesKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_calculateDerivatives()");
		}
	}

	void cl_initializeParameters(cl_command_queue queue, cl_mem params) const final
	{
		size_t globalSize = cl::DEFAULT_WORKGROUP_SIZE * outputSize;

		int error;
		error = clSetKernelArg(initKernel, 0, sizeof(cl_mem), &params);
		error |= clSetKernelArg(initKernel, 1, sizeof(inputSize), &inputSize);
		error |= clEnqueueNDRangeKernel(queue, initKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in layer::dense::cl_calculateDerivatives()");
		}
	}

	void cl_initKernels(cl_context context, cl_device_id device) final
	{
		// TODO: calculate workgroup size
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		forwardKernel = clCreateKernel(program, "forward", &error);
		backPropagateKernel = clCreateKernel(program, "backPropagate", &error);
		calculateDerivativesKernel = clCreateKernel(program, "calculateDerivatives", &error);
		initKernel = clCreateKernel(program, "initParams", &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for layer::dense.");
		}
	}

	const float* getBiases(const float* parameters) const { return parameters; }
	float* getBiases(float* parameters)       const { return parameters; }
	const float* getWeights(const float* parameters) const { return parameters + outputSize; }
	float* getWeights(float* parameters)       const { return parameters + outputSize; }

private:
	cl_kernel forwardKernel;

	cl_kernel backPropagateKernel;

	cl_kernel calculateDerivativesKernel;

	cl_kernel initKernel;
};
}
}