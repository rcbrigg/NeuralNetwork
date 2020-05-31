#include "..\include\network.hpp"
#include "network_data.hpp"
#include "host_impl.hpp"
#include "device_impl.hpp"

#include "layers\dense.hpp"
#include "layers\sigmoid.hpp"

#include "optimizers\sgd.hpp"
#include "optimizers\adam.hpp"

#include "losses\mse.hpp"
#include <iostream>

using namespace std;

namespace nn
{
Network::Network(NetworkArgs&& args)
{
	/// Validate config
	if (args.data->layers.size() == 0)
	{
		throw invalid_argument("Cannot create a network with 0 layers.");
	}

	if (args.data->cl)
	{
		if (cl::Wrapper::instance().init())
		{
			impl = make_unique<DeviceImpl>(move(args.data));
		}
		else
		{
			std::cout << "Could not create OpenCL context. Reverting to CPU implementation.\n";
			impl = make_unique<HostImpl>(move(args.data));
		}
	}
	else
	{
		impl = make_unique<HostImpl>(move(args.data));
	}

	if (!impl->init())
	{
		throw exception("Failed to create network.");
	}
}

Network::~Network()
{
}

Tensor<> Network::forward(ConstTensor<> inputs, size_t inputCount)
{
	return ((Impl*)impl.get())->forward(inputs, inputCount);
}

Tensor<1, uint32_t> Network::clasify(ConstTensor<> inputs, size_t inputCount)
{
	return impl->clasify(inputs, inputCount);
}

void Network::train(ConstTensor<> inputs, ConstTensor<> targets, size_t inputCount, uint32_t epochs, size_t batchSize)
{
	checkLossFunction();
	checkOptimizer();
	impl->train(inputs, targets, inputCount, batchSize, epochs);
}

void Network::train(ConstTensor<> inputs, Tensor<1, const uint32_t> targets, size_t inputCount, uint32_t epochs, size_t batchSize)
{
	checkOptimizer();
	impl->train(inputs, targets, inputCount, batchSize, epochs);
}

double Network::test(ConstTensor<> inputs, ConstTensor<> targets, size_t inputCount)
{
	checkLossFunction();
	return impl->test(inputs, targets, inputCount);
}

double Network::test(ConstTensor<> inputs, Tensor<1, const uint32_t> targets, size_t inputCount)
{
	return impl->test(inputs, targets, inputCount);
}

void Network::checkInputShape(Shape<> inputShape) const
{
	if (inputShape.slice() != impl->getConfig().inputShape)
	{
		throw std::invalid_argument("Input tensor shape does not match network configuration.");
	}
}

void Network::checkTargetShape(Shape<> inputShape, Shape<> targetShape) const
{
	if (targetShape.slice() != impl->getConfig().outputShape)
	{
		throw std::invalid_argument("Target tensor shape does not match network configuration.");
	}

	if (inputShape.length() != targetShape.length())
	{
		throw std::invalid_argument("Input and target tensor lengths do not match.");
	}
}

void Network::checkLabels(Shape<> inputShape, Shape<> targetShape) const
{
	if (inputShape.length() != targetShape.length())
	{
		throw std::invalid_argument("Input and target tensor lengths do not match.");
	}
}

void Network::checkLossFunction() const
{
	if (!impl->getConfig().lossFunc)
	{
		throw std::exception("No loss function has been set.");
	}
}

void Network::checkOptimizer() const
{
	if (!impl->getConfig().optimizer)
	{
		throw std::exception("No optimizer has been set.");
	}
}

NetworkArgs::NetworkArgs() :
	data(make_unique<NetworkConfig>())
{
}

NetworkArgs::~NetworkArgs()
{
}

void NetworkArgs::setInputShape(Shape<> shape)
{
	if (shape.size() == 0)
	{
		throw std::invalid_argument("Input shape cannot have size 0.");
	}
	if (data->layers.size() > 0)
	{
		throw std::invalid_argument("Input shape cannot be set after adding first layer.");
	}
	data->inputShape = shape;
	data->outputShape = shape;
}

void NetworkArgs::addLayerDense(uint32_t outputSize)
{
	checkAddLayer();

	if (outputSize == 0)
	{
		throw std::invalid_argument("Cannot have a layer with output size 0.");
	}

	const uint32_t inputSize = data->outputShape.size();
	auto layer = std::make_unique<layer::Dense>(inputSize, outputSize);
	data->layers.push_back(move(layer));
	data->outputShape = { outputSize };
}

void NetworkArgs::addLayerSigmoid()
{
	checkAddLayer();
	const uint32_t inputSize = data->outputShape.size();
	auto layer = std::make_unique<layer::Sigmoid>(inputSize);
	data->layers.push_back(move(layer));
}

void NetworkArgs::setOptimizerGradientDescent(float learningRate)
{
	auto optimizer = std::make_unique<optimizer::Sgd>(learningRate);
	data->optimizer = move(optimizer);
}

void NetworkArgs::setOptimizerAdam(float learningRate)
{
	auto optimizer = std::make_unique<optimizer::Adam>(learningRate);
	data->optimizer = move(optimizer);
}

void NetworkArgs::setLossMse()
{
	auto loss = std::make_unique<loss::Mse>();
	data->lossFunc = move(loss);
}

void NetworkArgs::enableOpenCLAcceleration(bool enable)
{
	data->cl = enable;
}

Shape<>& NetworkArgs::getOutputShape() const
{
	return data->outputShape;
}

void NetworkArgs::checkAddLayer()
{
	if (data->inputShape.size() == 0)
	{
		throw std::invalid_argument("Input size needs to be set before adding layers.");
	}
}
}