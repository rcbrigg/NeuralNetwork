#pragma once

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
		const float* input;
		const float* output;
		const float* outputError;
		const float* params;
	};

	virtual void backPropagate(const BackPropData& data, float* inputError) const  = 0;

	virtual void calculateDerivatives(const BackPropData& data, float* derivaitves) const {}

	virtual void initializeParameters(float* params) const {}

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