#include "pch.h"
#include "..\include\network.hpp"
#include "..\utils\utils.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace nn;

namespace test
{
namespace network
{
TEST_CLASS(Simple)
{
public:
	
	void Linear(bool cl)
	{
		auto makeSimpleNetwork = [=]()
		{
			NetworkArgs args;
			args.setInputShape({ 1 });
			args.addLayerDense(1);
			args.setLossMse();
			args.setOptimizerGradientDescent(0.1f);
			args.enableOpenCLAcceleration(cl);
			return Network(move(args));
		};

		auto f = [](float x)
		{
			return 3.0f * x - 1.5f;
		};

		auto network = makeSimpleNetwork();

		auto inputs = uniformRandomTensor(200, -2.f, 2.f).as<2>({ 200, 1 });
		auto targets = Tensor<2>({ inputs.size(), 1u });
		std::transform(inputs.data(), inputs.end(), targets.data(), f);

		network.train(inputs.section(0, 100), targets.section(0, 100), 1);
		auto error = network.test(inputs.section(100, 200), targets.section(100, 200));
		Assert::IsTrue(error < 0.00001f);
	}
	TEST_METHOD(Linear)
	{
		Linear(false);
	}
	TEST_METHOD(cl_Linear)
	{
		Linear(true);
	}

	void Parabola(bool cl)
	{
		auto makeSimpleNetwork = [=]()
		{
			NetworkArgs args;
			args.setInputShape({ 1 });
			args.addLayerDense(10);
			args.addLayerSigmoid();
			args.addLayerDense(1);
			args.setLossMse();
			args.setOptimizerGradientDescent(0.1f);
			args.enableOpenCLAcceleration(cl);
			return Network(move(args));
		};

		auto f = [](float x)
		{
			return 1.f - x * x;
		};

		auto network = makeSimpleNetwork();

		auto inputs = uniformRandomTensor(20000, -2.f, 2.f).as<2>({ 20000, 1 });
		auto targets = Tensor<2>({ inputs.size(), 1u });
		std::transform(inputs.data(), inputs.end(), targets.data(), f);

		network.train(inputs.section(0, 10000), targets.section(0, 10000), 10, 10);
		auto error = network.test(inputs.section(10000, 20000), targets.section(10000, 20000));
		Assert::IsTrue(error < 0.01f);
	}

	TEST_METHOD(Parabola)
	{
		Parabola(false);
	}

	TEST_METHOD(cl_Parabola)
	{
		Parabola(true);
	}
};
}
}