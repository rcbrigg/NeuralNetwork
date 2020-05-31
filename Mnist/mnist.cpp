#include "..\include\network.hpp"
#include "..\include\mnist_loader.hpp"
#include <iostream>
#include <chrono>

using namespace nn;
using namespace std;

Tensor<2> labelToVec(const Tensor<1, uint32_t>& labels)
{
	Tensor<2> result({ labels.size(), 10 });
	memset(result.data(), 0, result.size() * sizeof(float));
	for (size_t i = 0; i < labels.size(); ++i)
	{
		result.data()[10 * i + labels[i]] = 1.0f;
	}
	return result;

}

auto makeNetwork(Shape<> inputShape, size_t outputSize)
{
	NetworkArgs args;
	args.setInputShape(inputShape);
	args.addLayerDense(32);
	args.addLayerSigmoid();
	args.addLayerDense(outputSize);
	args.addLayerSigmoid();


	args.setLossMse();
	args.enableOpenCLAcceleration(true);


	//args.setOptimizerGradientDescent(3.0f);
	args.setOptimizerAdam(0.3f);
	return Network(move(args));
}

int main(int argc, char* argv[])
{
	MnistData data;

	try
	{
		//data = loadMnistData();
		data = LoadFormattedMnist();
	}
	catch (...)
	{
		cout << "Could not open mnist files." << endl;
		return 0;
	}

	//DumpFormattedMnist(data);

	auto network = makeNetwork(data.trainingData.shape().slice(), 10);


	cout << "Training network... ";

	auto start = std::chrono::high_resolution_clock::now();

	network.train(data.trainingData, data.trainingLabels, 1, 1);

	std::chrono::duration<float> elapsed = std::chrono::high_resolution_clock::now() - start;
	cout << "Done! (" << elapsed.count() << " seconds)" << endl;

	auto accuracy = (float)network.test(data.testData.section(0, 1000), data.testLabels.section(0, 1000));

	cout << "accuracy: " << accuracy << endl;

	return 0;
}