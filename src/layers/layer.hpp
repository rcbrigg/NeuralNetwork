#pragma once
#include "../cl/cl_utils.hpp"

namespace nn
{
namespace layer
{
class Layer
{
public:
	virtual void forward(const float* input, const float* params, float* output) const = 0;

	struct BackPropData
	{
		const float* input = nullptr;
		const float* output = nullptr;
		const float* outputError = nullptr;
		const float* params = nullptr;
	};

	virtual void backPropagate(const BackPropData& data, float* inputError) const  = 0;

	virtual void calculateDerivatives(const BackPropData& data, float* derivaitves) const {}

	virtual void initializeParameters(float* params) const {}

	struct ClBackPropData
	{
		cl_mem input = NULL;
		cl_mem output = NULL;
		cl_mem outputError = NULL;
		cl_mem params = NULL;
		uint32_t inputOffset  = 0;
	};

	virtual void cl_forward(cl_command_queue queue, cl_mem input, cl_mem params, cl_mem output, uint32_t inOffset, uint32_t outOffset, uint32_t paramOffset, uint32_t batchSize) const = 0;

	virtual void cl_backPropagate(cl_command_queue queue, const ClBackPropData& data, cl_mem inputError, uint32_t paramOffset, uint32_t batchSize) const = 0;

	virtual void cl_calculateDerivatives(cl_command_queue queue, const ClBackPropData& data, cl_mem derivaitves, uint32_t paramOffset, uint32_t batchSize) const {}

	virtual void cl_initializeParameters(cl_command_queue queue, cl_mem params, uint32_t offset) const {}

	virtual void cl_initKernels(cl_context context, cl_device_id device) {};

	const size_t getInputSize() const { return inputSize; }
	const size_t getOutputSize() const { return outputSize; }
	const size_t getParameterCount() const { return parmeterCount; }

protected:

	Layer(size_t inputSize, size_t outputSize, size_t parmeterCount) :
		inputSize(inputSize),
		outputSize(outputSize),
		parmeterCount(parmeterCount)
	{
		if (outputSize == 0)
		{
			throw std::invalid_argument("Output size of layer is 0.");
		}
	}

	const uint32_t inputSize;
	const uint32_t outputSize;
	const uint32_t parmeterCount;
};
}
}