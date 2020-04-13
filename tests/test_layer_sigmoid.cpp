#include "pch.h"
#include "..\src\layers\sigmoid.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace nn;
using namespace std;

namespace test
{
namespace layer
{
TEST_CLASS(Sigmoid)
{
public:

	TEST_METHOD(InvalidArgs)
	{
		auto invalidArgs = []
		{
			auto layer = nn::layer::Sigmoid(0);
		};
		Assert::ExpectException<std::invalid_argument>(invalidArgs);
	}

	TEST_METHOD(ForwardTest)
	{
		Tensor<> input = { 0.0f, -1.0f, 1.0f };
		Tensor<> output(input.size());
		auto layer = nn::layer::Sigmoid(input.size());
		layer.forward(input.data(), nullptr, output.data());
		for (size_t i = 0; i < input.size(); ++i)
		{
			Assert::AreEqual(layer.sigmoid(input[i]), output[i]);
		}
	}

	TEST_METHOD(BackpropTest)
	{
		Tensor<> input = { 0.0f, -1.0f, 1.0f };
		Tensor<> outputError = { 1.0f, 2.0f, 3.0f };
		Tensor<> inputError(input.size());
		nn::layer::Layer::BackPropData backProp;
		backProp.input = input.data();
		backProp.output = nullptr;
		backProp.outputError = outputError.data();
		backProp.params = nullptr;
		auto layer = nn::layer::Sigmoid(input.size());
		layer.backPropagate(backProp, inputError.data());
		for (size_t i = 0; i < input.size(); ++i)
		{
			Assert::AreEqual(layer.sigmoidPrime(input[i]) * outputError[i], inputError[i]);
		}
	}
};
}
}
