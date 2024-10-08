#include "include/gpu_utils.h"
#include "include/smp_utils.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <iostream>

namespace ppr::gpu
{
    void run_histogram_on_GPU(SOpenCLConfig& opencl, SConfig& configuration, SHistogram& hist, SDataStat& data_stat,
        double* data, long long begin, long long end, std::vector<int>& freq_buckets, double& var)
    {
        cl_int err = 0;
        const long long data_count = opencl.data_count_for_gpu;
        const unsigned long long work_group_number = data_count / opencl.wg_size;
        const unsigned long long count = data_count - (data_count % opencl.wg_size);

        // Result data
        double* out_var = new double[work_group_number];
        std::vector<cl_uint> out_histogram(2 * hist.binCount, 0);

        // Buffers
        cl::Buffer in_data_buf(opencl.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_USE_HOST_PTR, count * sizeof(double), data + begin, &err); 
        cl::Buffer out_sum_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, out_histogram.size() * sizeof(cl_uint), out_histogram.data(), &err);
        cl::Buffer out_var_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Set method arguments
        err = opencl.kernel.setArg(0, in_data_buf);
        err = opencl.kernel.setArg(1, opencl.wg_size * sizeof(double), nullptr);
        err = opencl.kernel.setArg(2, out_sum_buf);
        err = opencl.kernel.setArg(3, out_var_buf);
        err = opencl.kernel.setArg(4, sizeof(double), &data_stat.mean);
        err = opencl.kernel.setArg(5, sizeof(double), &data_stat.min);
        err = opencl.kernel.setArg(6, sizeof(double), &hist.scaleFactor);
        err = opencl.kernel.setArg(7, sizeof(double), &hist.binSize);
        err = opencl.kernel.setArg(8, sizeof(double), &hist.binCount);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Pass all data to GPU
        cl::CommandQueue cmd_queue(opencl.context, opencl.device, 0, &err);
        err = cmd_queue.enqueueNDRangeKernel(opencl.kernel, cl::NullRange, cl::NDRange(count), cl::NDRange(opencl.wg_size));

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Fill output vectors
        err = cmd_queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, out_histogram.size() * sizeof(cl_uint), out_histogram.data());
        err = cmd_queue.enqueueReadBuffer(out_var_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_var);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Wait on kernel
        cl::finish();

        // Agragate results
        int binCount = hist.binCount;
        for (int i = 0; i < binCount; i++)
        {
            const int value = out_histogram[2 * i] + out_histogram[2 * i + 1] * sizeof(cl_uint);
            freq_buckets[i] = freq_buckets[i] + value;
        }

        // Agregate results on CPU
        var += ppr::parallel::sum_vector_elements_vectorized(out_var, work_group_number);

        delete[] out_var;
    }

    void run_statistics_on_GPU(SDataStat& stat, SOpenCLConfig& m_ocl_config, SConfig& configuration, double* data, long long begin, long long end)
    {
        SDataStat local_stat;
        cl_int err = 0;
        const long long data_count = m_ocl_config.data_count_for_gpu;
        const unsigned long long work_group_number = data_count / m_ocl_config.wg_size;
        const unsigned long long count = data_count - (data_count % m_ocl_config.wg_size);

        cl::CommandQueue cmd_queue(m_ocl_config.context, m_ocl_config.device, 0, &err);

        // Result data
        double* out_sum = new double[work_group_number];
        double* out_min = new double[work_group_number];
        double* out_max = new double[work_group_number];

        // Input and output buffers
        cl::Buffer in_data_buf(m_ocl_config.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_USE_HOST_PTR, count * sizeof(double), data + begin, &err);
        cl::Buffer out_sum_buf(m_ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
        cl::Buffer out_min_buf(m_ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
        cl::Buffer out_max_buf(m_ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
        
        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Set method arguments
        err = m_ocl_config.kernel.setArg(0, in_data_buf);
        err = m_ocl_config.kernel.setArg(1, m_ocl_config.wg_size * sizeof(double), nullptr);
        err = m_ocl_config.kernel.setArg(2, m_ocl_config.wg_size * sizeof(double), nullptr);
        err = m_ocl_config.kernel.setArg(3, m_ocl_config.wg_size * sizeof(double), nullptr);
        err = m_ocl_config.kernel.setArg(4, out_sum_buf);
        err = m_ocl_config.kernel.setArg(5, out_min_buf);
        err = m_ocl_config.kernel.setArg(6, out_max_buf);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Pass all data to GPU
        err = cmd_queue.enqueueNDRangeKernel(m_ocl_config.kernel, cl::NullRange, cl::NDRange(count), cl::NDRange(m_ocl_config.wg_size));
        
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
        ppr::parallel::agregate_gpu_stat_vectorized(stat, out_sum, out_min, out_max, work_group_number);

        stat.n = data_count;

        delete[] out_sum;
        delete[] out_min;
        delete[] out_max;
    }

    void set_kernel_program(SOpenCLConfig& opencl, const std::string& file, const char* kernel_name)
    {
        // Update program
        create_kernel_program(opencl, file);

        // Update kernel
        create_kernel(opencl, kernel_name);
    }

    void create_kernel(SOpenCLConfig& opencl, const char* kernel_name)
    {
        cl_int err = 0;

        opencl.kernel = cl::Kernel(opencl.program, kernel_name, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        opencl.wg_size = opencl.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(opencl.device);
    }

    void create_kernel_program(SOpenCLConfig& opencl, const std::string& file)
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

        // Create first program
        opencl.context = cl::Context(opencl.device, nullptr, nullptr, nullptr, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        opencl.program = cl::Program(opencl.context, source, &err);

        if (err != CL_SUCCESS)
        {
            ppr::print_error(get_CL_error_string(err));
            return;
        }

        // Build our program
        err = opencl.program.build("-cl-std=CL2.0");

        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            // Get the build log for the first device
            std::string log = opencl.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(opencl.device);
            ppr::print_error(log);
        }
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
