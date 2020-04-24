#pragma once
//#include <cl/opencl.h>

namespace nn
{
namespace layer
{
class Layer
{
public:
	// Native methods
	struct BackPropData
	{
		const float* input;
		const float* output;
		const float* outputError;
		const float* params;
	};

	virtual void forward(const float* input, const float* params, float* output) const = 0;

	virtual void backPropagate(const BackPropData& data, float* inputError) const  = 0;

	virtual void calculateDerivatives(const BackPropData& data, float* derivaitves) const {}

	virtual void initializeParameters(float* params) const {}

	// OpenCl methods
	//struct OclBackPropData
	//{
	//	cl_mem input;
	//	cl_mem output;
	//	cl_mem outputError;
	//	cl_mem params;
	//};

	//virtual void OCL_forward(cl_command_queue queue,  cl_mem input, cl_mem params, cl_mem output) const = 0;

	//virtual void OCL_backPropagate(cl_command_queue queue, const OclBackPropData& data, cl_mem inputError) const = 0;

	//virtual void OCL_calculateDerivatives(cl_command_queue queue, const OclBackPropData& data, cl_mem derivaitves) const {}

	//virtual void OCL_initializeParameters(cl_command_queue queue, cl_mem params) const {}

	//virtual void OCL_createKernels(cl_context context, cl_device_id device) const {}

	// Get methods
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

	const size_t inputSize;
	const size_t outputSize;
	const size_t parmeterCount;
};
}
}