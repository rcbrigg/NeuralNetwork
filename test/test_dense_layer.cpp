#include "pch.h"
#include "..\src\layers\dense.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace nn;
using namespace nn::layer;
using namespace std;

namespace test
{
TEST_CLASS(DenseLayer)
{
public:

	TEST_METHOD(InvalidArgs)
	{
		auto invalidArgs = []
		{
			auto layer = Dense(0, 0);
		};
		Assert::ExpectException<std::invalid_argument>(invalidArgs);
	}

	TEST_METHOD(ConstructTest)
	{
		auto layer = Dense(3u, 4u);
		Assert::AreEqual(layer.getInputSize(), 3u);
		Assert::AreEqual(layer.getOutputSize(), 4u);
		Assert::AreEqual(layer.getParameterCount(), 3u * 4u + 4u);
	}

	TEST_METHOD(ForwardTest)
	{
		Tensor<> input = { -1.0f, 1.0f };
		Tensor<> output(2);
		auto layer = Dense(input.size(), output.size());
		auto params = Tensor<>(layer.getParameterCount());
		auto biases = layer.getBiases(params.data());
		auto weights = layer.getWeights(params.data());
		biases[0] = 2.0f;
		biases[1] = -1.0f;
		weights[0] = 1.0f;
		weights[1] = 3.0f;
		weights[2] = -2.0f;
		weights[3] = 4.0f;
		Tensor<> target = { 4.0f, 5.0f};
		layer.forward(input.data(), params.data(), output.data());
		Assert::IsTrue(target == output);
	}

	TEST_METHOD(BackpropTest)
	{
		Tensor<> inputError(2);
		Tensor<> outputError = { 3.0f, 4.0f };
		auto layer = Dense(inputError.size(), outputError.size());
		auto params = Tensor<>(layer.getParameterCount());
		auto biases = layer.getBiases(params.data());
		auto weights = layer.getWeights(params.data());
		biases[0] = 1.0f;
		biases[1] = 2.0f;
		weights[0] = -1.0f;
		weights[1] = 1.0f;
		weights[2] = 2.0f;
		weights[3] = -3.0f;
		
		
		Layer::BackPropData backProp;
		backProp.input = nullptr;
		backProp.output = nullptr;
		backProp.outputError = outputError.data();
		backProp.params = params.data();
		
		layer.backPropagate(backProp, inputError.data());
		Tensor<> expected = { 5.0f, -9.0f };

		for (size_t i = 0; i < inputError.size(); ++i)
		{
			Assert::AreEqual(inputError[i], expected[i]);
		}
	}

	TEST_METHOD(DerivativesTest)
	{
		Tensor<> input = { -1.0f, 1.0f };
		Tensor<> outputError = { 3.0f, 4.0f };
		Tensor<> dvs = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
		auto layer = Dense(input.size(), outputError.size());
				
		Layer::BackPropData backProp;
		backProp.input = input.data();
		backProp.output = nullptr;
		backProp.outputError = outputError.data();
		backProp.params = nullptr;

		layer.calculateDerivatives(backProp, dvs.data());
		auto db = layer.getBiases(dvs.data());
		auto dw = layer.getWeights(dvs.data());

		Assert::AreEqual(db[0], 3.0f);
		Assert::AreEqual(db[1], 4.0f);
		Assert::AreEqual(dw[0], -3.0f);
		Assert::AreEqual(dw[1], 3.0f);
		Assert::AreEqual(dw[2], -4.0f);
		Assert::AreEqual(dw[3], 4.0f);
	}
};
}
