#pragma once
#include "tensor.hpp"
#include "CL/opencl.h"

namespace nn
{
namespace optimizer
{
class Optimizer
{
public:
	// Perform any necessary data initialization
	virtual void init(float* derivatives, size_t paramCount) = 0;

	// Called at the start of each batch
	virtual void beginBatch(float* derivatives) = 0;

	// Update parameters after backpropagation pass (end of batch)
	virtual void update(float* parameters, const float* derivatives, size_t batchSize) = 0;

	// Perform any necessary data initialization
	virtual void cl_init(cl_context context, cl_device_id device, cl_command_queue queue, cl_mem derivatives, size_t paramCount) = 0;

	// Called at the start of each batch
	virtual void cl_beginBatch(cl_command_queue queue, cl_mem derivatives) = 0;

	// Update parameters after backpropagation pass (end of batch)
	virtual void cl_update(cl_command_queue queue, cl_mem parameters, cl_mem derivatives, size_t batchSize) = 0;
};
}
}
