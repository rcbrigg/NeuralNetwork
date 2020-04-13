#include "pch.h"
#include "..\src\layers\dense.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace nn;
using namespace std;

namespace test
{
namespace layer
{
TEST_CLASS(Dense)
{
public:

	TEST_METHOD(InvalidArgs)
	{
		auto invalidArgs = []
		{
			auto layer = nn::layer::Dense(0, 0);
		};
		Assert::ExpectException<std::invalid_argument>(invalidArgs);
	}

	TEST_METHOD(ConstructTest)
	{
		auto layer = nn::layer::Dense(3u, 4u);
		Assert::AreEqual(layer.getInputSize(), size_t(3));
		Assert::AreEqual(layer.getOutputSize(), size_t(4));
		Assert::AreEqual(layer.getParameterCount(), size_t(3 * 4 + 4));
	}

	TEST_METHOD(ForwardTest)
	{
		Tensor<> input = { -1.f, 1.f };
		Tensor<> output(2);
		auto layer = nn::layer::Dense(input.size(), output.size());
		auto params = Tensor<>(layer.getParameterCount());
		auto biases = layer.getBiases(params.data());
		auto weights = layer.getWeights(params.data());
		biases[0] = 2.f;
		biases[1] = -1.f;
		weights[0] = 1.f;
		weights[1] = 3.f;
		weights[2] = -2.f;
		weights[3] = 4.f;
		Tensor<> target = { 4.f, 5.f };
		layer.forward(input.data(), params.data(), output.data());
		Assert::IsTrue(target == output);
	}

	TEST_METHOD(BackpropTest)
	{
		Tensor<> inputError(2);
		Tensor<> outputError = { 3.f, 4.f };
		auto layer = nn::layer::Dense(inputError.size(), outputError.size());
		auto params = Tensor<>(layer.getParameterCount());
		auto biases = layer.getBiases(params.data());
		auto weights = layer.getWeights(params.data());
		biases[0] = 1.f;
		biases[1] = 2.f;
		weights[0] = -1.f;
		weights[1] = 1.f;
		weights[2] = 2.f;
		weights[3] = -3.f;


		nn::layer::Layer::BackPropData backProp;
		backProp.input = nullptr;
		backProp.output = nullptr;
		backProp.outputError = outputError.data();
		backProp.params = params.data();

		layer.backPropagate(backProp, inputError.data());
		Tensor<> expected = { 5.f, -9.f };

		for (size_t i = 0; i < inputError.size(); ++i)
		{
			Assert::AreEqual(expected[i], inputError[i]);
		}
	}

	TEST_METHOD(DerivativesTest)
	{
		Tensor<> input = { -1.f, 1.f };
		Tensor<> outputError = { 3.f, 4.f };
		Tensor<> dvs = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		auto layer = nn::layer::Dense(input.size(), outputError.size());

		nn::layer::Layer::BackPropData backProp;
		backProp.input = input.data();
		backProp.output = nullptr;
		backProp.outputError = outputError.data();
		backProp.params = nullptr;

		layer.calculateDerivatives(backProp, dvs.data());
		auto db = layer.getBiases(dvs.data());
		auto dw = layer.getWeights(dvs.data());

		Assert::AreEqual(3.f, db[0]);
		Assert::AreEqual(4.f, db[1]);
		Assert::AreEqual(-3.f, dw[0]);
		Assert::AreEqual(3.f, dw[1]);
		Assert::AreEqual(-4.f, dw[2]);
		Assert::AreEqual(4.f, dw[3]);
	}
};
}
}