#include "pch.h"
#include "..\src\layers\sigmoid.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace nn;
using namespace nn::layer;
using namespace std;

namespace test
{
TEST_CLASS(SigmoidLayer)
{
public:

	TEST_METHOD(InvalidArgs)
	{
		auto invalidArgs = []
		{
			auto layer = Sigmoid(0);
		};
		Assert::ExpectException<std::invalid_argument>(invalidArgs);
	}

	TEST_METHOD(ForwardTest)
	{
		Tensor<> input = { 0.0f, -1.0f, 1.0f };
		Tensor<> output(input.size());
		auto layer = Sigmoid(input.size());
		layer.forward(input.data(), nullptr, output.data());
		for (size_t i = 0; i < input.size(); ++i)
		{
			Assert::AreEqual(output[i], layer.sigmoid(input[i]));
		}
	}

	TEST_METHOD(BackpropTest)
	{
		Tensor<> input = { 0.0f, -1.0f, 1.0f };
		Tensor<> outputError = { 1.0f, 2.0f, 3.0f };
		Tensor<> inputError(input.size());
		Layer::BackPropData backProp;
		backProp.input = input.data();
		backProp.output = nullptr;
		backProp.outputError = outputError.data();
		backProp.params = nullptr;
		auto layer = Sigmoid(input.size());
		layer.backPropagate(backProp, inputError.data());
		for (size_t i = 0; i < input.size(); ++i)
		{
			Assert::AreEqual(inputError[i], layer.sigmoidPrime(input[i]) * outputError[i]);
		}
	}
};
}
