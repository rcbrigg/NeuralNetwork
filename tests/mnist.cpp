#include "pch.h"
#include "..\include\network.hpp"
#include "..\include\mnist_loader.hpp"
#include "..\utils\utils.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;
using namespace nn;

namespace test
{
namespace mnist
{
TEST_CLASS(Mnist)
{
public:
	enum class Optimizer
	{
		Sgd,
		Adam
	};

	auto makeNetwork(Shape<> inputShape, size_t outputSize, bool cl, Optimizer opt)
	{
		NetworkArgs args;
		args.setInputShape(inputShape);
		args.addLayerDense(64);
		args.addLayerSigmoid();
		args.addLayerDense(outputSize);
		args.addLayerSigmoid();
		args.setLossMse();
		args.enableOpenCLAcceleration(cl);

		switch (opt)
		{
		case Optimizer::Sgd: args.setOptimizerGradientDescent(1.0f); break;
		case Optimizer::Adam: args.setOptimizerAdam(0.3f); break;
		}

		return Network(move(args));
	}

	void TrainAndTest(bool cl, Optimizer opt)
	{
		auto data = LoadFormattedMnist();
		auto network = makeNetwork(data.trainingData.shape().slice(), 10, cl, opt);
		network.train(data.trainingData, data.trainingLabels, 16, 1);
		auto accuracy = (float)network.test(data.testData.section(0, 1000), data.testLabels.section(0, 1000));
		Assert::IsTrue(accuracy >= 0.9);
	}

	TEST_METHOD(Sgd)
	{
		TrainAndTest(false, Optimizer::Sgd);
	}
	TEST_METHOD(cl_Sgd)
	{
		TrainAndTest(true, Optimizer::Sgd);
	}

	TEST_METHOD(Adam)
	{
		TrainAndTest(false, Optimizer::Adam);
	}
	TEST_METHOD(cl_Adam)
	{
		TrainAndTest(true, Optimizer::Adam);
	}
};
}
}