#include "pch.h"
#include "cl_helper.hpp"

namespace cl
{

Helper::Helper()
{
	if (!nn::cl::Wrapper::instance().init())
	{
		throw std::exception();
	}

	int error;
	context = nn::cl::Wrapper::instance().getContext();
	device = nn::cl::Wrapper::instance().getDeviceId();
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	//auto program = nn::cl::buildProgramFromFile(__FILE__, context, device);
	//uploadKernel = clCreateKernel(program, "upload", &error);
	//downloadKernel = clCreateKernel(program, "download", &error);

	if (error != CL_SUCCESS)
	{
		throw std::exception();
	}
}

Helper::~Helper()
{
	for (auto& pair : buffers)
	{
		clReleaseMemObject(pair.first);
	}

	clReleaseCommandQueue(queue);
}

cl_mem Helper::makeBuffer(nn::Tensor<> data)
{
	int error;
	auto buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nn::cl::alignSize(data.size()) * sizeof(float), NULL,  &error);	
	error |= clEnqueueWriteBuffer(queue, buffer, true, 0, data.size() * sizeof(float), data.data(), 0, NULL, NULL);

	if (error != CL_SUCCESS)
	{
		throw std::exception();
	}

	buffers[buffer] = data.size();
	return buffer;
}

cl_mem Helper::makeBuffer(size_t size)
{
	int error;
	auto buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nn::cl::alignSize(size) * sizeof(float), NULL, &error);

	if (error != CL_SUCCESS)
	{
		throw std::exception();
	}

	buffers[buffer] = size;
	return buffer;
}

nn::Tensor<> Helper::getData(cl_mem buffer)
{
	const size_t size = buffers.at(buffer);
	nn::Tensor<> result(size);

	int error = clFlush(queue);
	error |= clEnqueueReadBuffer(queue, buffer, true, 0, size * sizeof(float), result.data(), 0, NULL, NULL);

	if (error != CL_SUCCESS)
	{
		throw std::exception();
	}

	return result;
}

//class Helper
//{
//public:
//	Helper();
//
//	~Helper();
//
//	cl_mem makeBuffer(size_t size);
//
//	cl_mem makeBuffer(nn::Tensor<> data);
//
//	nn::Tensor<> getData(cl_mem buffer);
//
//	auto cl_context getContext() const { return context; }
//
//	auto cl_context getQueue() const { return queue; }
//private:
//	cl_command_queue queue;
//
//	cl_context context;
//
//	std::unordered_map<cl_mem, size_t> buffers;
//};
}