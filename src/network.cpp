#include "..\include\network.hpp"
#include "network_data.hpp"
#include "host_impl.hpp"

using namespace std;

namespace nn
{

class Network::Impl
{
public:
	NetworkData data;

	HostImpl hostImpl;
};

Network::Network() :
	impl(new Impl())
{
}

Network::~Network()
{
	delete impl;
}

template<size_t N> void setInputSize(Shape shape);

template<size_t N> void setInputData(const Tensor<N>& data);

void addLayerDense(uint32_t outputSize);

void addLayerConv2d(uint32_t fileterCount, Dim2 filterSize, Dim2 filterStep = { 1, 1 });

void addLayerMaxPooling(Dim2 poolingSize);

void addLayerSigmoid();

void addLayerTanh();

void addLayerRelu();

void addLayerLrelu();

void setOptimizerGradientDescent(float learningRate = 0.01);

void setOptimizerAdam();

void Network::batchSize(uint32_t size)
{
	impl->data.batchSize = size;
}

void compile(bool gpu = true)
{

}

template<size_t N> void setTargets(const Tensor<>& data);

void setLabels(const Tensor<>& data)
{

}

void train(uint32_t epochs = 1)
{
}

Tensor<> Network::forward()
{
	impl->hostImpl.forward(impl->data);
}

Tensor<1, uint32_t> classify();
};
}