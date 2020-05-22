#include "cl_utils.hpp"
#include <exception>
#include <vector>
#include <iostream>

static cl_platform_id choosePlatform()
{
    // Get OpenCL platform count
    cl_uint numPlatforms;
    auto ciErrNum = clGetPlatformIDs(0, NULL, &numPlatforms);

    if (ciErrNum != CL_SUCCESS)
    {
        throw std::exception("OpenCL error in 'clGetPlatformIDs'!");
    }

    if (numPlatforms == 0)
    {
        throw std::exception("No OpenCL platform found!");
    }
    else
    {
        std::vector<cl_platform_id> platfromIds(numPlatforms);

        // get platform info for each platform and trap the NVIDIA platform if found
        ciErrNum = clGetPlatformIDs(numPlatforms, platfromIds.data(), NULL);

        if (ciErrNum != CL_SUCCESS)
        {
            throw std::exception("OpenCL error in 'clGetPlatformIDs'!");
        }

        for (auto id : platfromIds)
        {
            return id;
            //char buffer[1024];
            //ciErrNum = clGetPlatformInfo(id, CL_PLATFORM_NAME, 1024, &buffer, NULL);

            //if (ciErrNum == CL_SUCCESS)
            //{
            //    if (strstr(buffer, "NVIDIA") != NULL)
            //    {
            //        return id;
            //    }
            //}
        }

        // default to zeroeth platform if NVIDIA not found
        //std::cout << "WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!" << std::endl;
        return platfromIds[0];
    }
}

void printDeviceProperties(cl_device_id device)
{
    char device_string[1024];
    bool nv_device_attibute_query = false;

    // CL_DEVICE_NAME
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DEVICE_NAME: \t\t\t" << device_string << std::endl;

    // CL_DEVICE_VENDOR
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DEVICE_VENDOR:\t\t\t" << device_string << std::endl;

    // CL_DRIVER_VERSION
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DRIVER_VERSION:\t\t\t" << device_string << std::endl;

    // CL_DEVICE_VERSION
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_string), &device_string, NULL);
    std::cout << "CL_DEVICE_VERSION:\t\t\t" << device_string << std::endl;

    // CL_DEVICE_TYPE
    cl_device_type type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if (type & CL_DEVICE_TYPE_CPU)
        std::cout << "CL_DEVICE_TYPE:\t\t\t" << "CL_DEVICE_TYPE_CPU" << std::endl;
    if (type & CL_DEVICE_TYPE_GPU)
        std::cout << "CL_DEVICE_TYPE:\t\t\t\t" << "CL_DEVICE_TYPE_GPU" << std::endl;
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        std::cout << "CL_DEVICE_TYPE:\t\t\t" << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
    if (type & CL_DEVICE_TYPE_DEFAULT)
        std::cout << "CL_DEVICE_TYPE:\t\t\t" << "CL_DEVICE_TYPE_DEFAULT" << std::endl;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS:\t\t" << compute_units << std::endl;

    // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    size_t workitem_dims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t" << workitem_dims << std::endl;

    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t" << workitem_size[0] << workitem_size[1] << workitem_size[2] << std::endl;

    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t workgroup_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t" << workgroup_size << std::endl;

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t" << clock_frequency << "MHz" << std::endl;

    // CL_DEVICE_ADDRESS_BITS
    cl_uint addr_bits;
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
    std::cout << "CL_DEVICE_ADDRESS_BITS:\t\t\t" << addr_bits << std::endl;

    // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    cl_ulong max_mem_alloc_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t" << max_mem_alloc_size / (1024 * 1024) << "Mbyte" << std::endl;

    // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE:\t\t" << (unsigned int)(mem_size / (1024 * 1024)) << "MByte" << std::endl;

    // CL_DEVICE_ERROR_CORRECTION_SUPPORT
    cl_bool error_correction_support;
    clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
    std::cout << "CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t" << (error_correction_support == CL_TRUE ? "yes" : "no") << std::endl;

    // CL_DEVICE_LOCAL_MEM_TYPE
    cl_device_local_mem_type local_mem_type;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
    std::cout << "CL_DEVICE_LOCAL_MEM_TYPE:\t\t" << (local_mem_type == 1 ? "local" : "global") << std::endl;

    // CL_DEVICE_LOCAL_MEM_SIZE
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE:\t\t" << (unsigned int)(mem_size / 1024) << "KByte" << std::endl;

    // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t" << (unsigned int)(mem_size / 1024) << "KByte" << std::endl;

    // CL_DEVICE_QUEUE_PROPERTIES
    cl_command_queue_properties queue_properties;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
    if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        std::cout << "CL_DEVICE_QUEUE_PROPERTIES:\t\t" << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE" << std::endl;
    if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
        std::cout << "CL_DEVICE_QUEUE_PROPERTIES:\t\t" << "CL_QUEUE_PROFILING_ENABLE" << std::endl;

    // CL_DEVICE_IMAGE_SUPPORT
    cl_bool image_support;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
    std::cout << "CL_DEVICE_IMAGE_SUPPORT:\t\t" << image_support << std::endl;

    // CL_DEVICE_MAX_READ_IMAGE_ARGS
    cl_uint max_read_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
    std::cout << "CL_DEVICE_MAX_READ_IMAGE_ARGS:\t\t" << max_read_image_args << std::endl;

    // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
    cl_uint max_write_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
    std::cout << "CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t\t" << max_write_image_args << std::endl;

    // CL_DEVICE_SINGLE_FP_CONFIG
    cl_device_fp_config fp_config;
    clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp_config, NULL);
    std::cout << "CL_DEVICE_SINGLE_FP_CONFIG:\t\t" <<
        (fp_config & CL_FP_DENORM ? "denorms " : "") <<
        (fp_config & CL_FP_INF_NAN ? "INF-quietNaNs " : "") <<
        (fp_config & CL_FP_ROUND_TO_NEAREST ? "round-to-nearest " : "") <<
        (fp_config & CL_FP_ROUND_TO_ZERO ? "round-to-zero " : "") <<
        (fp_config & CL_FP_ROUND_TO_INF ? "round-to-inf " : "") <<
        (fp_config & CL_FP_FMA ? "fma " : "") << std::endl;

    // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
    size_t szMaxDims[5];
    std::cout << "CL_DEVICE_IMAGE <dim>" << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
    std::cout << "\t\t\t\t\t2D_MAX_WIDTH\t" << szMaxDims[0] << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
    std::cout << "\t\t\t\t\t2D_MAX_HEIGHT\t" << szMaxDims[1] << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
    std::cout << "\t\t\t\t\t3D_MAX_WIDTH\t" << szMaxDims[2] << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
    std::cout << "\t\t\t\t\t3D_MAX_HEIGHT\t" << szMaxDims[3] << std::endl;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
    std::cout << "\t\t\t\t\t3D_MAX_DEPTH\t" << szMaxDims[4] << std::endl;

    // CL_DEVICE_EXTENSIONS: get device extensions, and if any then parse & log the string onto separate lines
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_string), &device_string, NULL);
    if (device_string != 0)
    {
        std::cout << "CL_DEVICE_EXTENSIONS:" << std::endl;
        std::string stdDevString;
        stdDevString = std::string(device_string);
        size_t szOldPos = 0;
        size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
        while (szSpacePos != stdDevString.npos)
        {
            if (strcmp("cl_nv_device_attribute_query", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0)
                nv_device_attibute_query = true;

            //if (szOldPos > 0)
            {
                std::cout << "\t\t";
            }
            std::cout << "\t\t\t" << stdDevString.substr(szOldPos, szSpacePos - szOldPos) << std::endl;

            do {
                szOldPos = szSpacePos + 1;
                szSpacePos = stdDevString.find(' ', szOldPos);
            } while (szSpacePos == szOldPos);
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "CL_DEVICE_EXTENSIONS: None" << std::endl;
    }

    if (nv_device_attibute_query)
    {
        cl_uint compute_capability_major, compute_capability_minor;
        clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &compute_capability_major, NULL);
        clGetDeviceInfo(device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &compute_capability_minor, NULL);
        std::cout << "CL_DEVICE_COMPUTE_CAPABILITY_NV:\t" << compute_capability_major << "." << compute_capability_minor << std::endl;

        std::cout << "NUMBER OF MULTIPROCESSORS:\t\t" << compute_units << std::endl; // this is the same value reported by CL_DEVICE_MAX_COMPUTE_UNITS

        cl_uint regs_per_block;
        clGetDeviceInfo(device, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(cl_uint), &regs_per_block, NULL);
        std::cout << "CL_DEVICE_REGISTERS_PER_BLOCK_NV:\t" << regs_per_block << std::endl;

        cl_uint warp_size;
        clGetDeviceInfo(device, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &warp_size, NULL);
        std::cout << "CL_DEVICE_WARP_SIZE_NV:\t\t\t" << warp_size << std::endl;

        cl_bool gpu_overlap;
        clGetDeviceInfo(device, CL_DEVICE_GPU_OVERLAP_NV, sizeof(cl_bool), &gpu_overlap, NULL);
        std::cout << "CL_DEVICE_GPU_OVERLAP_NV:\t\t" << (gpu_overlap == CL_TRUE ? "CL_TRUE" : "CL_FALSE") << std::endl;

        cl_bool exec_timeout;
        clGetDeviceInfo(device, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &exec_timeout, NULL);
        std::cout << "CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:\t" << (exec_timeout == CL_TRUE ? "CL_TRUE" : "CL_FALSE") << std::endl;

        cl_bool integrated_memory;
        clGetDeviceInfo(device, CL_DEVICE_INTEGRATED_MEMORY_NV, sizeof(cl_bool), &integrated_memory, NULL);
        std::cout << "CL_DEVICE_INTEGRATED_MEMORY_NV:\t\t" << (integrated_memory == CL_TRUE ? "CL_TRUE" : "CL_FALSE") << std::endl;
    }
}

cl_device_id oclGetFirstDev(cl_platform_id platform)
{
    cl_device_id deviceId;
    cl_uint numDevices;

    auto ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1,
                                   &deviceId, &numDevices);

    if (ciErrNum != CL_SUCCESS || numDevices == 0)
    {
        throw std::exception("OpenCL error in 'clGetDeviceIDs'!");
    }

    return deviceId;
}

bool nn::cl::Wrapper::init()
{
    if (!initialized)
    {
        try
        {
            float zoomX = 1.0f;
            float zoomY = 1.0f;
            float centreX = 0.0f;
            float centreY = 0.0f;
            uint32_t width = 1024;
            uint32_t height = 1024;

            int err;

            auto platformId = choosePlatform();

            device = oclGetFirstDev(platformId);

            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

            if (err != CL_SUCCESS)
            {
                throw std::exception("Failed to create OpenCL context.");
            }

            //auto commands = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, &err);

            initialized = true;
            //clReleaseCommandQueue(commands);
        }
        catch (std::exception& e)
        {
            std::cout << e.what();
        }

       
    }

    return initialized;
}

void nn::cl::Wrapper::cleanUp()
{
    if (initialized)
    {
        clReleaseContext(context);
        initialized = false;
    }
}
