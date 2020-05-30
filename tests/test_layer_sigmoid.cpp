#include "pch.h"
#include "..\src\layers\sigmoid.hpp"
#include "..\utils\utils.hpp"

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
		backProp.outputError = outputError.data();
		auto layer = nn::layer::Sigmoid(input.size());
		layer.backPropagate(backProp, inputError.data());
		for (size_t i = 0; i < input.size(); ++i)
		{
			Assert::AreEqual(layer.sigmoidPrime(input[i]) * outputError[i], inputError[i]);
		}
	}
	TEST_METHOD(cl_ForwardTest)
	{
		auto input = nn::uniformRandomTensor(100, -5.f, 5.f).as<2>({ 5, 20 });
		auto target = Tensor<2>(input.shape());
		auto clInput = clHelper.makeBuffer(input.flat());
		auto clOutput = clHelper.makeBuffer(input.size());

		auto layer = nn::layer::Sigmoid(input.shape().size() / input.length());
		layer.cl_initKernels(clHelper.getContext(), clHelper.getDevice());
		layer.cl_forward(clHelper.getQueue(), clInput, nullptr, clOutput, 0, 0, 0, input.length());

		for (size_t i = 0; i < input.length(); i++)
		{
			layer.forward(input[i].data(), nullptr, target[i].data());
		}
		auto output = clHelper.getData(clOutput);
		Assert::IsTrue(areWithinTolerance(target.data(), output.data(), target.size(), 0.0001));
	}
	TEST_METHOD(cl_BackPropagateTest)
	{
		auto input = nn::uniformRandomTensor(100, -5.f, 5.f).as<2>({ 5, 20 });
		auto inputError = Tensor<2>(input.shape());
		auto outputError = nn::uniformRandomTensor(100, -5.f, 5.f).as<2>(inputError.shape());

		auto layer = nn::layer::Sigmoid(input.size() / input.length());

		auto clInput = clHelper.makeBuffer(input.flat());
		auto clInputError = clHelper.makeBuffer(inputError.size());
		auto clOutputError = clHelper.makeBuffer(outputError.flat());

		layer.cl_initKernels(clHelper.getContext(), clHelper.getDevice());

		nn::layer::Layer::ClBackPropData clBackProp;
		clBackProp.outputError = clOutputError;
		clBackProp.input = clInput;
		layer.cl_backPropagate(clHelper.getQueue(), clBackProp, clInputError, 0, input.length());

		nn::layer::Layer::BackPropData backProp;

		for (size_t i = 0; i < inputError.shape().length(); i++)
		{
			backProp.input = input[i].data();
			backProp.outputError = outputError[i].data();
			layer.backPropagate(backProp, inputError[i].data());
		}

		auto result = clHelper.getData(clInputError);

		Assert::IsTrue(areWithinTolerance(result.data(), inputError.data(), result.size(), 0.0001));
	}
private:
	::cl::Helper clHelper;
};
}
}
