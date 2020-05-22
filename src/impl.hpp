#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"
#include "../utils/utils.hpp"
#include "losses/loss.hpp"

namespace nn
{
class Impl
{
public:
	virtual bool init() = 0;

	virtual Tensor<> forward(const ConstTensor<>& input, size_t inputCount) = 0;

	virtual Tensor<1, uint32_t> clasify(const ConstTensor<>& input, size_t inputCount) = 0;

	virtual double test(const ConstTensor<>& input, const Tensor<1, const float>& targets, size_t inputCount) = 0;

	virtual double test(const ConstTensor<> & input, const Tensor<1, const uint32_t> & targets, size_t inputCount) = 0;

	virtual void train(const ConstTensor<>& inputs, const Tensor<1, const float>& targets, size_t inputCount) = 0;

	virtual void train(const ConstTensor<>& inputs, const Tensor<1, const uint32_t>& targets, size_t inputCount) = 0;

	const NetworkConfig& getConfig() const
	{
		return *config;
	}

protected:
	Impl(unique_ptr<const NetworkConfig>&& config) : config(move(config)) {}

	unique_ptr<const NetworkConfig> config;
};
}