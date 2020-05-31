#pragma once
#include "optimizer.hpp"

namespace nn
{
namespace optimizer
{
class Adam : public Optimizer
{
public:
	Adam(float learningRate) :
		learningRate(learningRate),
		parameterCount(0),
		initKernel(NULL),
		updateKernel(NULL),
		updateBetasKernel(NULL),
		mBuffer(NULL),
		vBuffer(NULL),
		betaPowBuffer(NULL),
		beta1Pow(0.f),
		beta2Pow(0.f)
	{
	}

	~Adam()
	{
		if (mBuffer)
		{
			clReleaseMemObject(mBuffer);
		}
		if (vBuffer)
		{
			clReleaseMemObject(vBuffer);
		}
		if (betaPowBuffer)
		{
			clReleaseMemObject(betaPowBuffer);
		}
	}

	void init(float* data, size_t paramCount) final
	{
		parameterCount = paramCount;
		beta1Pow = beta1;
		beta2Pow = beta2;
		t = 0;
		m = std::vector<float>(paramCount, 0.f);
		v = std::vector<float>(paramCount, 0.f);
	}

	void beginBatch(float* derivatives) final
	{
		memset(derivatives, 0, sizeof(float) * parameterCount);
	}

	void update(float* params, const float* derivatives, size_t batchSize) final
	{
		const float scale = learningRate / batchSize;

		for (size_t i = 0; i < parameterCount; ++i)
		{
			float g = derivatives[i];
			m[i] = beta1 * m[i] + (1 - beta1) * g;
			v[i] = beta2 * v[i] + (1 - beta2) * g * g;
			float mHat = m[i] / (1 - beta1Pow);
			float vHat = v[i] / (1 - beta2Pow);
			params[i] -= scale * mHat / (sqrtf(vHat) + epsilon);
		}

		beta1Pow *= beta1;
		beta2Pow *= beta2;
	}

	void cl_init(cl_context context, cl_device_id device, cl_command_queue, cl_mem, size_t paramCount) final
	{
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		updateKernel = clCreateKernel(program, "update", &error);
		updateBetasKernel = clCreateKernel(program, "updateBetas", &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for optimizer::adam.");
		}

		mBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, paramCount * sizeof(float), NULL, &error);
		vBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, paramCount * sizeof(float), NULL, &error);
		betaPowBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating buffers for optimizer::adam.");
		}

		parameterCount = paramCount;
	}

	void cl_beginTraining(cl_command_queue queue, cl_mem derivatives) final
	{
		float zero = 0.f;
		int error = clEnqueueFillBuffer(queue, derivatives, &zero, sizeof(zero), 0, parameterCount * sizeof(float), 0, NULL, NULL);
		error |= clEnqueueFillBuffer(queue, mBuffer, &zero, sizeof(zero), 0, parameterCount * sizeof(float), 0, NULL, NULL);
		error |= clEnqueueFillBuffer(queue, vBuffer, &zero, sizeof(zero), 0, parameterCount * sizeof(float), 0, NULL, NULL);

		float betas[] = { beta1, beta2 };
		error |= clEnqueueFillBuffer(queue, betaPowBuffer, betas, sizeof(betas), 0, sizeof(betas), 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::adam::cl_beginBatch()");
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
		error |= clSetKernelArg(updateKernel, 2, sizeof(cl_mem), &mBuffer);
		error |= clSetKernelArg(updateKernel, 3, sizeof(cl_mem), &vBuffer);
		error |= clSetKernelArg(updateKernel, 4, sizeof(cl_mem), &betaPowBuffer);
		error |= clSetKernelArg(updateKernel, 5, sizeof(scale), &scale);
		error |= clSetKernelArg(updateKernel, 6, sizeof(size), &size);
		error |= clEnqueueNDRangeKernel(queue, updateKernel, 1, NULL, &globalSize, &cl::workGroupSize, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::sgd::cl_update()");
		}

		error = clSetKernelArg(updateBetasKernel, 0, sizeof(cl_mem), &betaPowBuffer);
		globalSize = 1;
		error |= clEnqueueNDRangeKernel(queue, updateBetasKernel, 1, NULL, &globalSize, &globalSize, 0, NULL, NULL);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error in optimizer::sgd::cl_update()");
		}

		++t;
	}

private:
	float learningRate;

	size_t parameterCount;

	uint32_t t;

	std::vector<float> m, v;

	static constexpr float beta1 = 0.9f;
	static constexpr float beta2 = 0.999f;
	static constexpr float epsilon = 1e-8;

	float beta1Pow;
	float beta2Pow;

	cl_mem mBuffer;

	cl_mem vBuffer;

	cl_mem betaPowBuffer;

	cl_kernel updateKernel;

	cl_kernel updateBetasKernel;

	cl_kernel initKernel;
};
}
}
