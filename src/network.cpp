#include "..\include\network.hpp"
#include "network_data.hpp"
#include "host_impl.hpp"

#include "layers\dense.hpp"
#include "layers\sigmoid.hpp"

#include "optimizers\sgd.hpp"

#include "losses\mse.hpp"

using namespace std;

namespace nn
{
Network::Network(NetworkArgs&& args)
{
	/// Validate config
	if (args.data->layers.size() == 0)
	{
		throw invalid_argument("Cannot create a networkwith 0 layers");
	}

	impl = make_unique<HostImpl>(move(args.data));
}

Network::~Network()
{
}

Tensor<> Network::forward(ConstTensor<> inputs, size_t inputCount)
{
	return impl->forward(inputs, inputCount);
}

void Network::train(ConstTensor<> inputs, ConstTensor<> targets, size_t inputCount)
{
	impl->train(inputs, targets, inputCount);
}

void Network::checkInputShape(Shape<> inputShape)
{
	if (inputShape.slice() != impl->getConfig().inputShape)
	{
		throw std::invalid_argument("Input tensor shape does not match network configuration.");
	}
}

void Network::checkTrainingShape(Shape<> inputShape, Shape<> targetShape)
{
	if (inputShape.slice() != impl->getConfig().inputShape)
	{
		throw std::invalid_argument("Input tensor shape does not match network configuration.");
	}

	if (targetShape.slice() != impl->getConfig().outputShape)
	{
		throw std::invalid_argument("Target tensor shape does not match network configuration.");
	}

	if (inputShape.length() != targetShape.length())
	{
		throw std::invalid_argument("Input and target tensor lengths do not match.");
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

void NetworkArgs::setLossMse()
{
	auto loss = std::make_unique<loss::Mse>();
	data->lossFunc = move(loss);
}

void NetworkArgs::setBatchSize(uint32_t size)
{
	data->batchSize = size;
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