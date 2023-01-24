#pragma once

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_HPP_ENABLE_EXCEPTIONS


#undef min
#undef max

#include <CL/opencl.hpp>
#include <string>
#include "config.h"
#include "data.h"


namespace ppr::gpu
{

	/// <summary>
	/// Structure of opencl configuration
	/// </summary>
	struct SOpenCLConfig {
		cl::Device device{};							// Executing device
		cl::Context context{};                          // Opencl context
		cl::Program program{};                          // Opencl program
		cl::Kernel kernel{};                            // Opencl kernel          
		unsigned long long wg_size = 0;                 // One work group size
		unsigned long long data_count_for_gpu = 0;      // Data count for process on gpu
		unsigned long long data_count_for_cpu = 0;      // Data count for process on cpu
		unsigned long wg_count = 0;                     // Work group count
		DWORD high = 0; 
		DWORD low = 0;
		int thread_id = 0;
		DWORD64 data_count = 0;
	};

	bool init_opencl(cl::Device device, const std::string& file, const char* kernel_name, SOpenCLConfig& ocl_config);

	void run_statistics_on_device(SOpenCLConfig& ocl_config, SDataStat& stat, long long data_count, double* data);

	void run_histogram_on_device(SOpenCLConfig& ocl_config, SDataStat& stat, long long data_count, double* data,
		SHistogram& hist, std::vector<int>& freq_buckets, double& variance);

	/// <summary>
	/// Find all OpenCl devices
	/// </summary>
	/// <param name="platforms">- OpenCL platform</param>
	/// <param name="all_devices">- Vector of devices in a system</param>
	/// <param name="user_devices">- Vector of devices from user</param>
	void find_opencl_devices(std::vector<cl::Device>& all_devices, 
		std::vector<std::string>& user_devices);

	/// <summary>
	/// Getting CL error
	/// Source: https://gitlab.com/-/snippets/1958344
	/// </summary>
	/// <param name="error">- Error code</param>
	/// <returns></returns>
	std::string get_CL_error_string(cl_int error);
}