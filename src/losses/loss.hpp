#pragma once
#include "../cl/cl_utils.hpp"

namespace nn
{
namespace loss
{
class Loss
{
public:
	virtual void calculateError(const float* output, const float* target, float* error, size_t size) const = 0;

	virtual float calculateError(const float* output, const float* target, size_t size) const = 0;

	virtual void calculateDerivatives(const float* output, const float* target, float* derivatives, size_t size) const = 0;

	virtual void cl_calculateError(cl_command_queue queue, cl_mem output, cl_mem target, cl_mem error, uint32_t targetOffset, uint32_t size) const = 0;

	virtual void cl_calculateTotalError(cl_command_queue queue, cl_mem output, cl_mem target, cl_mem ouputError, uint32_t size, size_t count) const = 0;

	virtual void cl_calculateDerivatives(cl_command_queue queue, cl_mem output, cl_mem target, cl_mem derivatives, uint32_t targetOffset, uint32_t size) const = 0;

	virtual void cl_initKernels(cl_context context, cl_device_id device) {};
};
}
}
