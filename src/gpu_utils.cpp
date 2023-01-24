#include "include/gpu_utils.h"
#include "include/smp_utils.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <iostream>

namespace ppr::gpu
{

    bool init_opencl(cl::Device device, const std::string& file, const char* kernel_name, SOpenCLConfig& ocl_config)
    {
        cl_int err = 0;

        std::string file_path = __FILE__;
        std::string kernel_path = file_path.substr(0, file_path.rfind("\\")) + "\\" + file;

        // Read kernel from file
        std::ifstream kernel_file(kernel_path);
        std::string src((std::istreambuf_iterator<char>(kernel_file)), (std::istreambuf_iterator<char>()));
        const char* t_src = src.c_str();

        cl::Program::Sources source(1, { t_src, src.length() + 1 });
        kernel_file.close();

        // Set context
        ocl_config.context = cl::Context(device, nullptr, nullptr, nullptr, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return false;
        }

        ocl_config.program = cl::Program(ocl_config.context, source, &err);

        // Load kernel code
        ocl_config.program.build(device, "-cl-std=CL2.0");

        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            // Get the build log for the first device
            std::string log = ocl_config.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            ppr::print_error(log);
            return false;
        }

        // Set kernel
        ocl_config.kernel = cl::Kernel(ocl_config.program, kernel_name, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return false;
        }

        ocl_config.wg_size = ocl_config.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
        ocl_config.device = device;

        return true;
    }

    void run_histogram_on_device(SOpenCLConfig& ocl_config, SDataStat& stat, long long data_count, double* data,
        SHistogram& hist, std::vector<int>& freq_buckets, double& variance)
    {
        cl_int err = 0;

        const unsigned long long work_group_number = data_count / ocl_config.wg_size;
        const unsigned long long count = data_count - (data_count % ocl_config.wg_size);

        // Result data
        double* out_var = new double[work_group_number];
        cl_uint* out_histogram = new cl_uint[2 * hist.binCount] {0};

        //std::vector<double> out_var(work_group_number);
        //std::vector<cl_uint> out_histogram(2 * hist.binCount, 0);

        // Buffers
        cl::Buffer in_data_buf(ocl_config.context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, count * sizeof(double), data, &err);
        cl::Buffer out_sum_buf(ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2 * hist.binCount * sizeof(cl_uint), out_histogram, &err);
        cl::Buffer out_var_buf(ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Set method arguments
        err = ocl_config.kernel.setArg(0, in_data_buf);
        err = ocl_config.kernel.setArg(1, ocl_config.wg_size * sizeof(double), nullptr);
        err = ocl_config.kernel.setArg(2, out_sum_buf);
        err = ocl_config.kernel.setArg(3, out_var_buf);
        err = ocl_config.kernel.setArg(4, sizeof(double), &stat.mean);
        err = ocl_config.kernel.setArg(5, sizeof(double), &stat.min);
        err = ocl_config.kernel.setArg(6, sizeof(double), &hist.scaleFactor);
        err = ocl_config.kernel.setArg(7, sizeof(double), &hist.binSize);
        err = ocl_config.kernel.setArg(8, sizeof(double), &hist.binCount);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Pass all data to GPU
        cl::CommandQueue cmd_queue(ocl_config.context, ocl_config.device, 0, &err);
        err = cmd_queue.enqueueNDRangeKernel(ocl_config.kernel, cl::NullRange, cl::NDRange(count), cl::NDRange(ocl_config.wg_size));

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Fill output vectors
        err = cmd_queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, 2 * hist.binCount * sizeof(cl_uint), out_histogram);
        err = cmd_queue.enqueueReadBuffer(out_var_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_var);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Wait on kernel
        cl::finish();

        // Agragate results
        for (int i = 0; i < hist.binCount; i++)
        {
            const int value = out_histogram[2 * i] + out_histogram[2 * i + 1] * sizeof(cl_uint);
            freq_buckets[i] += value;
        }

        // Agregate results on CPU
        variance += ppr::parallel::sum_vector_elements_vectorized(out_var, 2 * hist.binCount);
    }

    void run_statistics_on_device(SOpenCLConfig& ocl_config, SDataStat& stat, long long data_count, double* data)
    {
        cl_int err = 0;
        
        const unsigned long long work_group_number = data_count / ocl_config.wg_size;
        const unsigned long long count = data_count - (data_count % ocl_config.wg_size);

        cl::CommandQueue cmd_queue(ocl_config.context, ocl_config.device, 0, &err);

        // Input and output buffers
        cl::Buffer in_data_buf(ocl_config.context, CL_MEM_READ_ONLY  | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, count * sizeof(double), data, &err);
        cl::Buffer out_sum_buf(ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
        cl::Buffer out_min_buf(ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
        cl::Buffer out_max_buf(ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Set method arguments
        err = ocl_config.kernel.setArg(0, in_data_buf);
        err = ocl_config.kernel.setArg(1, ocl_config.wg_size * sizeof(double), nullptr);
        err = ocl_config.kernel.setArg(2, ocl_config.wg_size * sizeof(double), nullptr);
        err = ocl_config.kernel.setArg(3, ocl_config.wg_size * sizeof(double), nullptr);
        err = ocl_config.kernel.setArg(4, out_sum_buf);
        err = ocl_config.kernel.setArg(5, out_min_buf);
        err = ocl_config.kernel.setArg(6, out_max_buf);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Result data
        double* out_sum = new double[work_group_number];
        double* out_min = new double[work_group_number];
        double* out_max = new double[work_group_number];

        // Pass all data to GPU
        err = cmd_queue.enqueueNDRangeKernel(ocl_config.kernel, cl::NullRange, cl::NDRange(count), cl::NDRange(ocl_config.wg_size));

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Fill output vectors
        err = cmd_queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_sum);
        err = cmd_queue.enqueueReadBuffer(out_min_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_min);
        err = cmd_queue.enqueueReadBuffer(out_max_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_max);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        cl::finish();

        // Agregate results on CPU
        double sum = ppr::parallel::sum_vector_elements_vectorized(out_sum, work_group_number);
        double max = ppr::parallel::max_of_vector_vectorized(out_max, work_group_number);
        double min = ppr::parallel::min_of_vector_vectorized(out_min, work_group_number);

        stat.sum = sum;
        stat.max = max;
        stat.min = min;
        stat.n = data_count;
        //stat.mean = ;
        //stat.variance = ;
        //stat.isNegative = ;

    }

    void find_opencl_devices(std::vector<cl::Device>& all_devices, std::vector<std::string>& user_devices)
    {
        // Find all platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (const auto& platform : platforms)
        {
            // Get all devices of the current platform.
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

            for (const auto& device : devices)
            {
                bool available = device.getInfo<CL_DEVICE_AVAILABLE>();
                std::string device_name = device.getInfo<CL_DEVICE_NAME>();
                std::string device_extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
                bool exists = std::binary_search(user_devices.begin(), user_devices.end(), device_name);

                if (available &&
                    device_extensions.find("cl_khr_fp64") != std::string::npos &&
                    (user_devices.size() == 0 || std::find(user_devices.begin(), user_devices.end(), device_name) != user_devices.end()))
                {
                    all_devices.emplace_back(device);
                }
            }
        }
    }


    std::string get_CL_error_string(cl_int error)
    {
        switch (error) {
            // run-time and JIT compiler errors
            case 0: return "CL_SUCCESS";
            case -1: return "CL_DEVICE_NOT_FOUND";
            case -2: return "CL_DEVICE_NOT_AVAILABLE";
            case -3: return "CL_COMPILER_NOT_AVAILABLE";
            case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5: return "CL_OUT_OF_RESOURCES";
            case -6: return "CL_OUT_OF_HOST_MEMORY";
            case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8: return "CL_MEM_COPY_OVERLAP";
            case -9: return "CL_IMAGE_FORMAT_MISMATCH";
            case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11: return "CL_BUILD_PROGRAM_FAILURE";
            case -12: return "CL_MAP_FAILURE";
            case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case -15: return "CL_COMPILE_PROGRAM_FAILURE";
            case -16: return "CL_LINKER_NOT_AVAILABLE";
            case -17: return "CL_LINK_PROGRAM_FAILURE";
            case -18: return "CL_DEVICE_PARTITION_FAILED";
            case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
            case -30: return "CL_INVALID_VALUE";
            case -31: return "CL_INVALID_DEVICE_TYPE";
            case -32: return "CL_INVALID_PLATFORM";
            case -33: return "CL_INVALID_DEVICE";
            case -34: return "CL_INVALID_CONTEXT";
            case -35: return "CL_INVALID_QUEUE_PROPERTIES";
            case -36: return "CL_INVALID_COMMAND_QUEUE";
            case -37: return "CL_INVALID_HOST_PTR";
            case -38: return "CL_INVALID_MEM_OBJECT";
            case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40: return "CL_INVALID_IMAGE_SIZE";
            case -41: return "CL_INVALID_SAMPLER";
            case -42: return "CL_INVALID_BINARY";
            case -43: return "CL_INVALID_BUILD_OPTIONS";
            case -44: return "CL_INVALID_PROGRAM";
            case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46: return "CL_INVALID_KERNEL_NAME";
            case -47: return "CL_INVALID_KERNEL_DEFINITION";
            case -48: return "CL_INVALID_KERNEL";
            case -49: return "CL_INVALID_ARG_INDEX";
            case -50: return "CL_INVALID_ARG_VALUE";
            case -51: return "CL_INVALID_ARG_SIZE";
            case -52: return "CL_INVALID_KERNEL_ARGS";
            case -53: return "CL_INVALID_WORK_DIMENSION";
            case -54: return "CL_INVALID_WORK_GROUP_SIZE";
            case -55: return "CL_INVALID_WORK_ITEM_SIZE";
            case -56: return "CL_INVALID_GLOBAL_OFFSET";
            case -57: return "CL_INVALID_EVENT_WAIT_LIST";
            case -58: return "CL_INVALID_EVENT";
            case -59: return "CL_INVALID_OPERATION";
            case -60: return "CL_INVALID_GL_OBJECT";
            case -61: return "CL_INVALID_BUFFER_SIZE";
            case -62: return "CL_INVALID_MIP_LEVEL";
            case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64: return "CL_INVALID_PROPERTY";
            case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
            case -66: return "CL_INVALID_COMPILER_OPTIONS";
            case -67: return "CL_INVALID_LINKER_OPTIONS";
            case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
            case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
            case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
            case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
            case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
            case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
            case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
            default: return "Unknown OpenCL error";
        }


    }
}
