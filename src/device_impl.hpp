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
		uploadInputsDefaultKernel(NULL),
		downloadOutputsKernel(NULL),
		classifyOutputsKernel(NULL),
		calcClassifyDerivativesKernel(NULL)
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

		context = cl::Wrapper::instance().getContext();
		device = cl::Wrapper::instance().getDeviceId();
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);

		if (error != CL_SUCCESS)
		{
			return false;
		}

		inputStride = cl::alignSize(config->inputShape.size());
		outputStride = cl::alignSize(config->outputShape.size());

		size_t parametersSize = 0;
		for (const auto& layer : config->layers)
		{
			layer->cl_initKernels(context, device);
			parametersSize += cl::alignSize(layer->getParameterCount());
		}

		
		parameters = clCreateBuffer(context, CL_MEM_READ_WRITE, parametersSize * sizeof(float), NULL, &error);
		derivatives = clCreateBuffer(context, CL_MEM_READ_WRITE, parametersSize * sizeof(float), NULL, &error);

		size_t paramOffset = 0;

		for (const auto& layer : config->layers)
		{
			size_t size = cl::alignSize(layer->getParameterCount()) * sizeof(float);
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
			size_t alignedOutputSize = cl::alignSize(layer->getOutputSize()) * sizeof(float);
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

	Tensor<> forward(const ConstTensor<>& input, size_t inputCount) final
	{
		createBuffer(inputBuffer, inputStride, inputCount);
		createBuffer(outputBuffer, outputStride, inputCount);
		uploadData(input.data(), inputBuffer, inputCount, config->inputShape.size());

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(inputBuffer, outputBuffer, i * inputStride, i * outputStride);
		}

		const size_t outputsSize = inputCount * config->layers.back()->getOutputSize();
		outputs = Tensor<>(outputsSize);
		downloadOutputData(outputs, inputCount);
		return outputs;
	}

	Tensor<1, uint32_t> clasify(const ConstTensor<>& input, size_t inputCount) final
	{
		createBuffer(inputBuffer, inputStride, inputCount);
		createBuffer(outputBuffer, 1, inputCount);
		uploadData(input.data(), inputBuffer, inputCount, config->inputShape.size());

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(inputBuffer, outputBuffer, i * inputStride, i * outputStride);
		}

		classifications = Tensor<1, uint32_t>(inputCount);
		classifyOutputData();

		return classifications;
	}

	double test(const ConstTensor<>& input, const Tensor<1, const uint32_t>& targets, size_t inputCount) final
	{
		createBuffer(inputBuffer, inputStride, inputCount);
		createBuffer(outputBuffer, outputStride, inputCount);
		uploadData(input.data(), inputBuffer, inputCount, config->inputShape.size());
		int error;
		if (targetBuffer)
		{
			clReleaseMemObject(targetBuffer);
		}

		targetBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputCount * sizeof(uint32_t), (void*)targets.data(), &error);

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(inputBuffer, outputBuffer, i * inputStride, i * outputStride);
		}

		classifications = Tensor<1, uint32_t>(inputCount);

		// TODO, also calculate loss
		classifyOutputData();

		double accuracy = 0.0;
		for (size_t i = 0; i < classifications.size(); ++i)
		{
			accuracy += classifications[i] == targets[i];
		}

		return accuracy / double(inputCount);
	}

	double test(const ConstTensor<>& input, const Tensor<1, const float>& targets, size_t inputCount) final
	{
		createBuffer(inputBuffer, inputStride, inputCount);
		createBuffer(outputBuffer, outputStride, inputCount);
		createBuffer(targetBuffer, outputStride, inputCount);
		uploadData(input.data(), inputBuffer, inputCount, config->inputShape.size());
		uploadData(targets.data(), targetBuffer, inputCount, config->outputShape.size());

		for (size_t i = 0; i < inputCount; ++i)
		{
			forward(inputBuffer, outputBuffer, i * inputStride, i * outputStride);
		}

		int error = 0;
		auto data = clEnqueueMapBuffer(queue, targetBuffer, CL_TRUE, CL_MAP_READ, 0, 3200 * sizeof(float), 0, NULL, NULL, &error);
		auto data1 = clEnqueueMapBuffer(queue, outputBuffer, CL_TRUE, CL_MAP_READ, 0, 3200 * sizeof(float), 0, NULL, NULL, &error);

		cl_mem errorBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputCount * sizeof(float), NULL, &error);
		config->lossFunc->cl_calculateTotalError(queue, outputBuffer, targetBuffer, errorBuffer, config->outputShape.size(), inputCount);
		Tensor<> errors(inputCount);
		error |= clEnqueueReadBuffer(queue, errorBuffer, CL_TRUE, 0, inputCount * sizeof(float), errors.data(), 0, NULL, NULL);
		clReleaseMemObject(errorBuffer);

		//auto data = clEnqueueMapBuffer(queue, targetBuffer, CL_TRUE, CL_MAP_READ, 0, 3200 * sizeof(float), 0, NULL, NULL, &error);
		//auto data1 = clEnqueueMapBuffer(queue, outputBuffer, CL_TRUE, CL_MAP_READ, 0, 3200 * sizeof(float), 0, NULL, NULL, &error);
		//auto data2 = clEnqueueMapBuffer(queue, parameters, CL_TRUE, CL_MAP_READ, 0, 2000 * sizeof(float), 0, NULL, NULL, &error);

		double loss = 0;
		for (size_t i = 0; i < errors.size(); ++i)
		{
			loss += errors[i];
		}

		return loss / double(inputCount);
	}

	void train(const ConstTensor<>& inputs, const Tensor<1, const float>& targets, size_t inputCount) final
	{
		createBuffer(inputBuffer, inputStride, inputCount);
		createBuffer(outputBuffer, outputStride, inputCount);
		createBuffer(targetBuffer, outputStride, inputCount);
		uploadData(inputs.data(), inputBuffer, inputCount, config->inputShape.size());
		uploadData(targets.data(), targetBuffer, inputCount, config->outputShape.size());

		for (size_t i = 0; i < inputCount;)
		{
			size_t batchEnd = std::min(i + config->batchSize, inputCount);
			size_t batchSize = batchEnd - i;
			config->optimizer->cl_beginBatch(queue, derivatives);

			for (; i < batchEnd; ++i)
			{
				train<false>(inputBuffer, targetBuffer, i);
			}

			config->optimizer->cl_update(queue, parameters, derivatives, batchSize);
		}
	}

	void train(const ConstTensor<>& inputs, const Tensor<1, const uint32_t>& targets, size_t inputCount) final
	{
		int error;
		createBuffer(inputBuffer, inputStride, inputCount);
		createBuffer(outputBuffer, outputStride, inputCount);
		if (targetBuffer)
		{
			clReleaseMemObject(targetBuffer);
		}

		targetBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputCount * sizeof(uint32_t), (void*)targets.data(), &error);
		uploadData(inputs.data(), inputBuffer, inputCount, config->inputShape.size());

		for (size_t i = 0; i < inputCount;)
		{
			size_t batchEnd = std::min(i + config->batchSize, inputCount);
			size_t batchSize = batchEnd - i;
			config->optimizer->cl_beginBatch(queue, derivatives);

			for (; i < batchEnd; ++i)
			{
				train<true>(inputBuffer, targetBuffer, i);
			}

			config->optimizer->cl_update(queue, parameters, derivatives, batchSize);
		}
	}

private:
	void uploadData(const void* data, cl_mem buffer, size_t count, const uint32_t size)
	{
		const uint32_t stride = cl::alignSize(size);
		const uint32_t hostSize = size * count; // TODO: 64 bit 
		const size_t totalStride = count * stride;
		int error;

		auto tempBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, hostSize * sizeof(float), (void*)data, &error);
		error = clSetKernelArg(uploadInputsDefaultKernel, 0, sizeof(cl_mem), &tempBuffer);
		error |= clSetKernelArg(uploadInputsDefaultKernel, 1, sizeof(cl_mem), &buffer);
		error |= clSetKernelArg(uploadInputsDefaultKernel, 2, sizeof(size), &size);
		error |= clSetKernelArg(uploadInputsDefaultKernel, 3, sizeof(stride), &stride);
		error |= clSetKernelArg(uploadInputsDefaultKernel, 4, sizeof(hostSize), &hostSize);
		error |= clEnqueueNDRangeKernel(queue, uploadInputsDefaultKernel, 1, NULL, &totalStride, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);
		error |= clReleaseMemObject(tempBuffer);
		
		if (error != CL_SUCCESS)
		{
			throw std::exception("Upload data failed.");
		}
	}

	void createBuffer(cl_mem& buffer, size_t stride, size_t count)
	{
		if (buffer)
		{
			clReleaseMemObject(buffer);
		}

		int error;
		const size_t totalSize = count * stride * sizeof(float);

		buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, totalSize, NULL, &error);

		if (error != CL_SUCCESS)
		{
			throw std::exception("Error: failed to allocate GPU memory.");
		}
	}

	void downloadOutputData(const Tensor<>& hostMem, size_t count)
	{
		int error;

		const uint32_t outputSize = config->outputShape.size();;
		const uint32_t outputStride = cl::alignSize(outputSize);
		const uint32_t hostSize = hostMem.size(); // TODO: 64 bit 
		const size_t globalSize = cl::alignSize(hostMem.size());

		auto tempBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,  hostSize * sizeof(float), (void*)hostMem.data(), &error);
		error = clSetKernelArg(downloadOutputsKernel, 0, sizeof(cl_mem), &outputBuffer);
		error |= clSetKernelArg(downloadOutputsKernel, 1, sizeof(cl_mem), &tempBuffer);
		error |= clSetKernelArg(downloadOutputsKernel, 2, sizeof(outputSize), &outputSize);
		error |= clSetKernelArg(downloadOutputsKernel, 3, sizeof(outputStride), &outputStride);
		error |= clSetKernelArg(downloadOutputsKernel, 4, sizeof(hostSize), &hostSize);
		error |= clEnqueueNDRangeKernel(queue, downloadOutputsKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);
		clFlush(queue);
		auto data = clEnqueueMapBuffer(queue, tempBuffer, CL_TRUE, CL_MAP_READ, 0, hostSize * sizeof(float), 0, NULL, NULL, & error);
		clEnqueueUnmapMemObject(queue, tempBuffer, data, 0, NULL, NULL);
		clReleaseMemObject(tempBuffer);
	}

	void classifyOutputData()
	{
		int error;

		const uint32_t outputSize = config->outputShape.size();
		const uint32_t outputStride = cl::alignSize(outputSize);

		auto tempBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, classifications.size() * sizeof(uint32_t), (void*)classifications.data(), &error);

		error = clSetKernelArg(classifyOutputsKernel, 0, sizeof(cl_mem), &outputBuffer);
		error |= clSetKernelArg(classifyOutputsKernel, 1, sizeof(cl_mem), &tempBuffer);
		error |= clSetKernelArg(classifyOutputsKernel, 2, sizeof(outputStride), &outputStride);
		error |= clSetKernelArg(classifyOutputsKernel, 3, sizeof(outputSize), &outputSize);
		const size_t globalSize = cl::DEFAULT_WORKGROUP_SIZE * classifications.size();

		error | clEnqueueNDRangeKernel(queue, classifyOutputsKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);

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
			layers[i]->cl_forward(queue, layerInput, layerParams[i], layerOutputs[i], inputOffset, 0);
			layerInput = layerOutputs[i];
			inputOffset = 0;
		}

		layers.back()->cl_forward(queue, layerInput, layerParams.back(), output, inputOffset, outputOffset);
	}

	template<bool CLASSIFY>
	void train(cl_mem input, cl_mem target, size_t index)
	{
		const size_t inputOffset = index * inputStride;

		forward(input, layerOutputs.back(), inputOffset, 0);

		const size_t layerCount = config->layers.size();

		calculateOutputDerivatives<CLASSIFY>(layerOutputs.back(), target, config->outputShape.size(), layerError.back(), index);

		//if (index == 50000)
		//{
		//	int error;
		//	auto data = clEnqueueMapBuffer(queue, targetBuffer, CL_TRUE, CL_MAP_READ, 50000 * 4, 100, 0, NULL, NULL, &error);
		//	auto data1 = clEnqueueMapBuffer(queue, layerError.back(), CL_TRUE, CL_MAP_READ, 0, 128, 0, NULL, NULL, &error);
		//	auto data2 = clEnqueueMapBuffer(queue, layerOutputs.back(), CL_TRUE, CL_MAP_READ, 0, 128, 0, NULL, NULL, &error);
		//	printf("g");
		//}
		layer::Layer::ClBackPropData data;

		for (size_t i = layerCount - 1; i > 0; --i)
		{
			layer::Layer& layer = *config->layers[i];

			cl_mem inputError = layerError[i - 1];			
			data.input = layerOutputs[i - 1];
			data.params = layerParams[i];
			data.output = layerOutputs[i];
			data.outputError = layerError[i];
			layer.cl_backPropagate(queue, data, inputError);
			layer.cl_calculateDerivatives(queue, data, layerDerivatives[i]);
		}

		data.input = input;
		data.params = layerParams.front();
		data.output = layerOutputs.front();
		data.outputError = layerError.front();
		data.inputOffset = inputOffset;
		config->layers[0]->cl_calculateDerivatives(queue, data, layerDerivatives.front());
	}

	template<bool CLASSIFY>
	void calculateOutputDerivatives(cl_mem output, cl_mem target, uint32_t outputSize, cl_mem outputError, uint32_t index) const
	{
		int error = clSetKernelArg(calcClassifyDerivativesKernel, 0, sizeof(cl_mem), &output);
		error |= clSetKernelArg(calcClassifyDerivativesKernel, 1, sizeof(cl_mem), &target);
		error |= clSetKernelArg(calcClassifyDerivativesKernel, 2, sizeof(cl_mem), &outputError);
		error |= clSetKernelArg(calcClassifyDerivativesKernel, 3, sizeof(index), &index);
		error |= clSetKernelArg(calcClassifyDerivativesKernel, 4, sizeof(outputSize), &outputSize);
		size_t globalSize = outputStride;
		error |= clEnqueueNDRangeKernel(queue, calcClassifyDerivativesKernel, 1, NULL, &globalSize, &cl::DEFAULT_WORKGROUP_SIZE, 0, NULL, NULL);
	}

	template<>
	void calculateOutputDerivatives<false>(cl_mem output, cl_mem target, uint32_t outputSize, cl_mem outputError, uint32_t index) const
	{
		return config->lossFunc->cl_calculateDerivatives(queue, output, target, outputError, index * outputStride, outputSize);
	}

	void initKernels()
	{
		auto program = cl::buildProgramFromFile(__FILE__, context, device);

		int error;
		uploadInputsDefaultKernel = clCreateKernel(program, "uploadInputDataDefault", &error);
		downloadOutputsKernel = clCreateKernel(program, "downloadOutputData", &error);
		classifyOutputsKernel = clCreateKernel(program, "classifyOutputDataDefault", &error);
		calcClassifyDerivativesKernel = clCreateKernel(program, "classifierCalcDerivatives", &error);

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

	cl_kernel uploadInputsDefaultKernel;

	cl_kernel downloadOutputsKernel;

	cl_kernel classifyOutputsKernel;

	cl_kernel calcClassifyDerivativesKernel;

	uint32_t inputStride;

	uint32_t outputStride;

};
}