#pragma once
#include "network_data.hpp"
#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"
#include "../utils/utils.hpp"
#include "losses/loss.hpp"
#include "cl/cl_utils.hpp"
#include "impl.hpp"

namespace nn
{
class DeviceImpl : public Impl
{
public:
	DeviceImpl(unique_ptr<const NetworkConfig>&& config) :
		Impl(move(config)),
		inputBuffer(NULL),
		outputBuffer(NULL),
		targetBuffer(NULL),
		parameters(NULL),
		derivatives(NULL),
		queue(NULL),
		context(NULL),
		device(0),
		classifyKernel(NULL),
		softmaxErrorKernel(NULL)
	{
	}

	~DeviceImpl()
	{
		releaseBuffers();
		clReleaseCommandQueue(queue);
	}

	bool init() final
	{
		int error;

		// Get CL objects
		context = cl::Wrapper::instance().getContext();
		device = cl::Wrapper::instance().getDeviceId();
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);

		if (error != CL_SUCCESS)
		{
			return false;
		}

		size_t parametersSize = 0;
		for (const auto& layer : config->layers)
		{
			// Setup kernels
			layer->cl_initKernels(context, device);

			// parameters for each layer start a new cacheline
			parametersSize += layer->getParameterCount();
		}

		// Create 1 buffer each for parameters and derivatives
		parameters = clCreateBuffer(context, CL_MEM_READ_WRITE, parametersSize * sizeof(float), NULL, &error);
		derivatives = clCreateBuffer(context, CL_MEM_READ_WRITE, parametersSize * sizeof(float), NULL, &error);

		size_t paramOffset = 0;

		// Create sub-buffers for each layer
		for (const auto& layer : config->layers)
		{
			size_t size = layer->getParameterCount() * sizeof(float);
			size_t range[2] = { paramOffset, size };
			paramOffset += size;

			if (size != 0)
			{
				layerParams.push_back(clCreateSubBuffer(parameters, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, range, &error));
				layerDerivatives.push_back(clCreateSubBuffer(derivatives, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, range, &error));
			}
			else
			{
				layerParams.push_back(nullptr);
				layerDerivatives.push_back(nullptr);
			}
			size_t alignedOutputSize = layer->getOutputSize() * sizeof(float);
			layerError.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, alignedOutputSize, NULL, &error));
			layerOutputs.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, alignedOutputSize, NULL, &error));

			layer->cl_initializeParameters(queue, layerParams.back());
		}

		if (config->optimizer)
		{
			config->optimizer->cl_init(context, device, queue, derivatives, parametersSize);
		}

		if (config->lossFunc)
		{
			config->lossFunc->cl_initKernels(context, device);
		}

		if (error != CL_SUCCESS)
		{
			return false;
		}

		try
		{
			initKernels();
		}
		catch (...)
		{
			return false;
		}

		return true;
	}

	Tensor<> forward(const ConstTensor<>& inputs, size_t inputCount) final
	{
		forwardCommon(inputs.data(), inputCount);
		const size_t outputsSize = inputCount * config->outputShape.size();
		outputs = Tensor<>(outputsSize);
		readOutputData(outputs.data(), inputCount);
		return outputs;
	}

	Tensor<1, uint32_t> clasify(const ConstTensor<>& inputs, size_t inputCount) final
	{
		forwardCommon(inputs.data(), inputCount);
		classifications = Tensor<1, uint32_t>(inputCount);
		classifyOutputData();
		return classifications;
	}

	double test(const ConstTensor<>& inputs, const Tensor<1, const uint32_t>& targets, size_t inputCount) final
	{
		forwardCommon(inputs.data(), inputCount);

		createBuffer(targetBuffer, (void*)targets.data(), 1, inputCount);

		classifications = Tensor<1, uint32_t>(inputCount);

		classifyOutputData();
		//int error;
		//auto data = clEnqueueMapBuffer(queue, targetBuffer, CL_TRUE, CL_MAP_READ, 0, 100 * sizeof(float), 0, NULL, NULL, &error);
		//auto data1 = clEnqueueMapBuffer(queue, outputBuffer, CL_TRUE, CL_MAP_READ, 0, 1000 * sizeof(float), 0, NULL, NULL, &error);
		//auto data2 = clEnqueueMapBuffer(queue, parameters, CL_TRUE, CL_MAP_READ, 0, 100 * sizeof(float), 0, NULL, NULL, &error);

		double accuracy = 0.0;
		for (size_t i = 0; i < classifications.size(); ++i)
		{
			accuracy += classifications[i] == targets[i];
		}

		return accuracy / double(inputCount);
	}

	double test(const ConstTensor<>& inputs, const Tensor<1, const float>& targets, size_t inputCount) final
	{
		forwardCommon(inputs.data(), inputCount);

		createBuffer(targetBuffer, (void*)targets.data(), config->outputShape.size(), inputCount);

		int error = 0;
		cl_mem errorBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputCount * sizeof(float), NULL, &error);

		//auto data = clEnqueueMapBuffer(queue, targetBuffer, CL_TRUE, CL_MAP_READ, 0, 100 * sizeof(float), 0, NULL, NULL, &error);
		//auto data1 = clEnqueueMapBuffer(queue, outputBuffer, CL_TRUE, CL_MAP_READ, 0, 100 * sizeof(float), 0, NULL, NULL, &error);
		//auto data2 = clEnqueueMapBuffer(queue, parameters, CL_TRUE, CL_MAP_READ, 0, 2 * sizeof(float), 0, NULL, NULL, &error);
		config->lossFunc->cl_calculateTotalError(queue, outputBuffer, targetBuffer, errorBuffer, inputCount, config->outputShape.size());

		Tensor<> errors(inputCount);

		error |= clEnqueueReadBuffer(queue, errorBuffer, CL_TRUE, 0, inputCount * sizeof(float), errors.data(), 0, NULL, NULL);
		clReleaseMemObject(errorBuffer);

		double loss = 0;
		for (size_t i = 0; i < errors.size(); ++i)
		{
			loss += errors[i];
		}

		return loss / double(inputCount);
	}

	void train(const ConstTensor<>& inputs, const Tensor<1, const float>& targets, size_t inputCount) final
	{
		trainCommon<false>(inputs.data(), targets.data(), inputCount);
	}

	void train(const ConstTensor<>& inputs, const Tensor<1, const uint32_t>& targets, size_t inputCount) final
	{
		trainCommon<true>(inputs.data(), targets.data(), inputCount);
	}

private:
	void createBuffer(cl_mem& buffer, void* data, uint32_t width, uint32_t height)
	{
		int error;

		const size_t allocSize = height * width * sizeof(float);

		// Allocate or grow buffer if needed
		if (buffer)
		{
			size_t currentSize;
			size_t retSize;
			error = clGetMemObjectInfo(buffer,
									   CL_MEM_SIZE,
									   sizeof(currentSize),
									   &currentSize,
									   &retSize);

			if (currentSize < allocSize)
			{
				error |= clReleaseMemObject(buffer);
				buffer = NULL;
			}

			if (error) throw std::exception();
		}

		if (!buffer)
		{
			buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
		}
		
		if (error) throw std::exception("Failed to allocate memory.");

		if (data)
		{
			error = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, allocSize, data, 0, NULL, NULL);
			if (error) throw std::exception();
		}
	}

	void forwardCommon(const float* input, size_t inputCount)
	{
		createBuffer(inputBuffer, (void*)input, config->inputShape.size(), inputCount);
		createBuffer(outputBuffer, nullptr, config->outputShape.size(), inputCount);

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(inputBuffer, outputBuffer, i * config->inputShape.size(), i * config->outputShape.size());
		}
	}

	template<bool Classify>
	void trainCommon(const void* inputs, const void* targets, size_t inputCount) 
	{
		createBuffer(inputBuffer, (void*)inputs, config->inputShape.size(), inputCount);
		createBuffer(outputBuffer, nullptr, config->outputShape.size(), inputCount);
		createBuffer(targetBuffer, (void*)targets, Classify ? 1 : config->outputShape.size(), inputCount);

		config->optimizer->cl_beginTraining(queue, derivatives);

		for (size_t i = 0; i < inputCount;)
		{
			size_t batchEnd = std::min(i + config->batchSize, inputCount);
			size_t batchSize = batchEnd - i;
			
			for (; i < batchEnd; ++i)
			{
				train<Classify>(inputBuffer, targetBuffer, i);
			}

			config->optimizer->cl_update(queue, parameters, derivatives, batchSize);
		}
	}

	void readOutputData(void* data, size_t count)
	{
		int error = clEnqueueReadBuffer(queue,
										outputBuffer,
										CL_FALSE,
										0,
										count * config->outputShape.size() * sizeof(float),
										data,
										0,
										NULL,
										NULL);

		if (error) throw std::exception();
	}

	void classifyOutputData()
	{
		int error;

		const uint32_t outputSize = config->outputShape.size();
		const uint32_t outputStride = outputSize;

		auto tempBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, classifications.size() * sizeof(uint32_t), (void*)classifications.data(), &error);

		error = clSetKernelArg(classifyKernel, 0, sizeof(cl_mem), &outputBuffer);
		error |= clSetKernelArg(classifyKernel, 1, sizeof(cl_mem), &tempBuffer);
		error |= clSetKernelArg(classifyKernel, 2, sizeof(outputStride), &outputStride);
		error |= clSetKernelArg(classifyKernel, 3, sizeof(outputSize), &outputSize);
		const size_t globalSize = cl::DEFAULT_WORKGROUP_SIZE * classifications.size();

		error | clEnqueueNDRangeKernel(queue, classifyKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

		clFlush(queue);
		auto data = clEnqueueMapBuffer(queue, tempBuffer, CL_TRUE, CL_MAP_READ, 0, classifications.size() * sizeof(uint32_t), 0, NULL, NULL, &error);
		clEnqueueUnmapMemObject(queue, tempBuffer, data, 0, NULL, NULL);

		clReleaseMemObject(tempBuffer);
	}

	void releaseBuffers()
	{
		clReleaseMemObject(inputBuffer);
		clReleaseMemObject(outputBuffer);
		clReleaseMemObject(targetBuffer);
	}

	using Layers = vector<unique_ptr<layer::Layer>>;

	void forward(cl_mem input, cl_mem output, uint32_t inputOffset, uint32_t outputOffset)
	{
		cl_mem layerInput = input;
		const auto& layers = config->layers;

		for (size_t i = 0; i < layers.size() - 1; ++i)
		{
			layers[i]->cl_forward(queue, layerInput, layerParams[i], layerOutputs[i], inputOffset, 0, 1);
			layerInput = layerOutputs[i];
			inputOffset = 0;
		}

		layers.back()->cl_forward(queue, layerInput, layerParams.back(), output, inputOffset, outputOffset, 1);
	}

	template<bool CLASSIFY>
	void train(cl_mem input, cl_mem target, size_t index)
	{
		const size_t inputOffset = index * config->inputShape.size();

		forward(input, layerOutputs.back(), inputOffset, 0);

		const size_t layerCount = config->layers.size();

		calculateOutputDerivatives<CLASSIFY>(layerOutputs.back(), target, config->outputShape.size(), layerError.back(), index);

		layer::Layer::ClBackPropData data;

		for (size_t i = layerCount - 1; i > 0; --i)
		{
			layer::Layer& layer = *config->layers[i];

			cl_mem inputError = layerError[i - 1];			
			data.input = layerOutputs[i - 1];
			data.params = layerParams[i];
			data.output = layerOutputs[i];
			data.outputError = layerError[i];
			layer.cl_backPropagate(queue, data, inputError, 1);
			layer.cl_calculateDerivatives(queue, data, layerDerivatives[i], 1);
		}

		data.input = input;
		data.params = layerParams.front();
		data.output = layerOutputs.front();
		data.outputError = layerError.front();
		data.inputOffset = inputOffset;
		config->layers[0]->cl_calculateDerivatives(queue, data, layerDerivatives.front(), 1);
	}

	template<bool CLASSIFY>
	void calculateOutputDerivatives(cl_mem output, cl_mem target, uint32_t outputSize, cl_mem outputError, uint32_t index) const
	{
		uint32_t size = outputSize * 1;
		size_t globalSize = cl::alignSize(size);

		int error = clSetKernelArg(softmaxErrorKernel, 0, sizeof(cl_mem), &output);
		error |= clSetKernelArg(softmaxErrorKernel, 1, sizeof(cl_mem), &target);
		error |= clSetKernelArg(softmaxErrorKernel, 2, sizeof(cl_mem), &outputError);
		error |= clSetKernelArg(softmaxErrorKernel, 3, sizeof(index), &index);
		error |= clSetKernelArg(softmaxErrorKernel, 4, sizeof(outputSize), &outputSize);
		error |= clSetKernelArg(softmaxErrorKernel, 5, sizeof(size), &size);
		
		error |= clEnqueueNDRangeKernel(queue, softmaxErrorKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);
	}

	template<>
	void calculateOutputDerivatives<false>(cl_mem output, cl_mem target, uint32_t outputSize, cl_mem outputError, uint32_t index) const
	{
		return config->lossFunc->cl_calculateDerivatives(queue,
														 output,
														 target,
														 outputError,
														 index * config->outputShape.size(),
														 1,
														 outputSize);
	}

	void initKernels()
	{
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		classifyKernel = clCreateKernel(program, "classify", &error);
		softmaxErrorKernel = clCreateKernel(program, "softmaxError", &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Unexpected error while creating kernel(s) for layer::cl_initKernels().");
		}
	}

	template<typename T> size_t getTargetSize() { return config->layers.back()->getOutputSize(); }

	template<> size_t getTargetSize<uint32_t>() { return 1; }

	cl_mem inputBuffer;

	cl_mem outputBuffer;

	cl_mem targetBuffer;

	// outputs of each layer
	std::vector<cl_mem> layerOutputs;

	// error of each layer output during backpropagation
	std::vector<cl_mem> layerError;

	// parameter gradients of each layer
	std::vector<cl_mem> layerDerivatives;

	std::vector<cl_mem> layerParams;

	// outputs of final layer
	Tensor<> outputs;

	// maximum output of final layer
	Tensor<1, uint32_t> classifications;

	// network weights and biases
	cl_mem parameters;

	// data used by optimiser (e.g. derivatives)
	cl_mem derivatives;

	cl_command_queue queue;

	cl_context context;

	cl_device_id device;

	cl_kernel classifyKernel;

	cl_kernel softmaxErrorKernel;
};
}