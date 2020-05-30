#include "pch.h"
#include "..\src\losses\mse.hpp"
#include "..\utils\utils.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace nn;

namespace test
{
namespace loss
{
TEST_CLASS(Mse)
{
public:

	TEST_METHOD(Error)
	{
		float output[] = { 0.f, 1.f, 2.0f };
		float target[] = { -1.f, 1.f, 5.f };
		float expected[] = { 1.f, 0.f, 9.f };
		float error[3];

		nn::loss::Mse lossFunc;
		lossFunc.calculateError(output, target, error, 3);
		for (size_t i = 0; i < 3; ++i)
		{
			Assert::AreEqual(expected[i], error[i]);
		}
		
		auto loss = lossFunc.calculateError(output, target, 3);
		Assert::AreEqual(10.f, loss);
	}

	TEST_METHOD(Derivatives)
	{
		float output[] = { 0.f, 1.f, 2.0f };
		float target[] = { -1.f, 1.f, 5.f };
		float expected[] = { 2.f, 0.f, -6.f };
		float error[3];

		nn::loss::Mse lossFunc;
		lossFunc.calculateDerivatives(output, target, error, 3);
		for (size_t i = 0; i < 3; ++i)
		{
			Assert::AreEqual(expected[i], error[i]);
		}
	}

	TEST_METHOD(cl_Error)
	{
		nn::loss::Mse lossFunc;
		lossFunc.cl_initKernels(clHelper.getContext(), clHelper.getDevice());
		auto output = uniformRandomTensor(100, -5.f, 5.f);
		auto target = uniformRandomTensor(output.size(), -5.f, 5.f);
		auto error = Tensor<>(output.size());
		lossFunc.calculateError(output.data(), target.data(), error.data(), output.size());

		auto clOutput = clHelper.makeBuffer(output);
		auto clTarget = clHelper.makeBuffer(target);
		auto clError = clHelper.makeBuffer(error.size());
		lossFunc.cl_calculateError(clHelper.getQueue(), clOutput, clTarget, clError, 0, output.size(), 1);
		auto result = clHelper.getData(clError);

		Assert::IsTrue(areWithinTolerance(error.data(), result.data(), error.size(), 0.0001));
	}

	TEST_METHOD(cl_TotalError)
	{
		nn::loss::Mse lossFunc;
		lossFunc.cl_initKernels(clHelper.getContext(), clHelper.getDevice());
		auto output = uniformRandomTensor(128, -5.f, 5.f);
		auto target = uniformRandomTensor(output.size(), -5.f, 5.f);
		auto clOutput = clHelper.makeBuffer(output);
		auto clTarget = clHelper.makeBuffer(target);
		auto clError = clHelper.makeBuffer(4);

		auto error = Tensor<>(output.size());
		lossFunc.calculateError(output.data(), target.data(), error.data(), output.size());

		Tensor<> errors(4);

		for (size_t i = 0; i < 4; ++i)
		{		
			errors[i] = 0;
			for (size_t j = 0; j < 32; ++j)
			{
				errors[i] += error[j + 32 * i];
			}
		}
		lossFunc.cl_calculateTotalError(clHelper.getQueue(), clOutput, clTarget, clError, 4, 32);
		auto result = clHelper.getData(clError);

		Assert::IsTrue(areWithinTolerance(errors.data(), result.data(), errors.size(), 0.001));
	}

	TEST_METHOD(cl_Derivatives)
	{
		nn::loss::Mse lossFunc;
		lossFunc.cl_initKernels(clHelper.getContext(), clHelper.getDevice());
		auto output = uniformRandomTensor(100, -5.f, 5.f);
		auto target = uniformRandomTensor(output.size(), -5.f, 5.f);
		auto error = Tensor<>(output.size());
		lossFunc.calculateDerivatives(output.data(), target.data(), error.data(), output.size());

		auto clOutput = clHelper.makeBuffer(output);
		auto clTarget = clHelper.makeBuffer(target);
		auto clError = clHelper.makeBuffer(error.size());
		lossFunc.cl_calculateDerivatives(clHelper.getQueue(), clOutput, clTarget, clError, 0, output.size(), 1);
		auto result = clHelper.getData(clError);

		Assert::IsTrue(areWithinTolerance(error.data(), result.data(), error.size(), 0.0001));
	}

private:
	::cl::Helper clHelper;
};
}
}