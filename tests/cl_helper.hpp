#pragma once
#include "..\src\cl\cl_utils.hpp"
#include "..\include\tensor.hpp"
#include <unordered_map>
namespace cl
{
class Helper
{
public:
	Helper();

	~Helper();

	cl_mem makeBuffer(size_t size);

	cl_mem makeBuffer(nn::Tensor<> data);

	nn::Tensor<> getData(cl_mem buffer);

	auto getContext() const { return context; }

	auto getQueue() const { return queue; }

	auto getDevice() const { return device; }

private:
	cl_command_queue queue;

	cl_context context;

	cl_device_id device;

	//cl_kernel uploadKernel;

	//cl_kernel downloadKernel;

	std::unordered_map<cl_mem, size_t> buffers;
};
}