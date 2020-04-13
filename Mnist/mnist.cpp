#include "..\include\network.hpp"
#include "mnist_loader.hpp"
#include <iostream>

using namespace nn;
using namespace std;

auto makeNetwork(Shape<> inputShape, size_t outputSize)
{
	NetworkArgs args;
	args.setInputShape(inputShape);
	args.addLayerDense(30);
	args.addLayerSigmoid();
	args.addLayerDense(outputSize);
	args.addLayerSigmoid();
	//args.setLossMse();
	args.setOptimizerGradientDescent(3.0f);
	args.setBatchSize(10);
	return Network(move(args));
}

int main(int argc, char* argv[])
{
	MnistData data;

	try
	{
		data = loadMnistData();
	}
	catch (...)
	{
		cout << "Could not open mnist files." << endl;
		return 0;
	}

	auto network = makeNetwork(data.trainingData.shape().slice(), 10);

	cout << "Training network... ";
	network.train(data.trainingData, data.trainingLabels);
	cout << "Done!" << endl;

	auto accuracy = (float)network.test(data.testData.section(0, 1000), data.testLabels.section(0, 1000));

	cout << "accuracy: " << accuracy << endl;

	return 0;
}