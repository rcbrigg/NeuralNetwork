#pragma once
#include "CL/opencl.h"
#include <stdio.h>
#include <string>
#include <iostream>

namespace nn
{
namespace cl
{
inline cl_program buildProgramFromFile(const char* fileName, cl_context context, cl_device_id device)
{
	int error = CL_SUCCESS;

    std::string  clName(fileName);
    size_t extPos = clName.find_first_of('.');
    clName.erase(extPos + 1);
    clName.append("cl");
    
    FILE* fp = NULL;
    fopen_s(&fp, clName.c_str(), "r+b");

    if (fp == NULL)
    {
        auto msg = std::string("Connot fild file ") + clName;
        throw std::exception(msg.c_str());
    }

    fseek(fp, 0, SEEK_END);
    size_t kernelSize = ftell(fp);
    rewind(fp);

    auto src = (char*)malloc(kernelSize);
    fread(src, sizeof(char), kernelSize, fp);
    fclose(fp);

    auto program = clCreateProgramWithSource(context, 1, (const char**)&src, &kernelSize, &error);

    if (error != CL_SUCCESS)
    {
        throw std::exception("Unexpected creating program");
    }

    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (error != CL_SUCCESS)
    {
        size_t len;
        char buffer[40000];

        std::cout << "Error: Failed to build program executable!\n";
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << buffer << std::endl;

        throw std::exception("Unexpected error building program");
    }

    return program;
}

static const size_t DEFAULT_WORKGROUP_SIZE = 32;

inline size_t alignSize(size_t s)
{
    return (s + 0x1F) & ~0x1F;
}

inline size_t getStride(size_t s)
{
    if (s > 32)
    {
        return (s + 0x1F) & ~0x1F;
    }
    else if (s > 16)
    {
        return 32;
    }
    else if (s > 8)
    {
        return 16;
    }
    else if (s > 4)
    {
        return 8;
    }
    else if (s > 2)
    {
        return 4;
    }
    else if (s > 1)
    {
        return 2;
    }
    else
    {
        return 1;
    }
}

class Wrapper
{
public:
    static auto& instance()
    {
        static Wrapper inst;
        return inst;
    }

    bool init();

    void cleanUp();

    auto getDeviceId() const { return device; }

    auto getContext() const { return context; }

    ~Wrapper()
    {
        cleanUp();
    }

private:
    Wrapper() = default;

    bool initialized = false;

    cl_context context = NULL;

    cl_device_id device = NULL;
};
}
}
