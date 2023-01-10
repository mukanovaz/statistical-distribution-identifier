#pragma once

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_HPP_ENABLE_EXCEPTIONS

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
		cl::Device device{};						// Executing device
		cl::Context context{};                          // Opencl context
		cl::Program program{};                          // Opencl program
		cl::Kernel kernel{};                            // Opencl kernel          
		unsigned long long wg_size = 0;                 // One work group size
		unsigned long long data_count_for_gpu = 0;      // Data count for process on gpu
		unsigned long long data_count_for_cpu = 0;      // Data count for process on cpu
		unsigned long wg_count = 0;                     // Work group count
	};

	/// <summary>
	/// Create data histogram on Opencl device
	/// </summary>
	/// <param name="opencl">- Opencl configuration structure</param>
	/// <param name="configuration">- Program configuration structure</param>
	/// <param name="hist">- Histogram configuration structure</param>
	/// <param name="data_stat">- Final Data statistics structure</param>
	/// <param name="data">- Data pointer</param>
	/// <param name="data_count">- Data count to process</param>
	/// <param name="freq_buckets">- Frequency histogram reference</param>
	/// <param name="var">- Final Variance</param>
	void run_histogram_on_GPU(SOpenCLConfig& opencl, SConfig& configuration, SHistogram& hist, SDataStat& data_stat,
		double* data, long long begin, long long end, std::vector<int>& freq_buckets, double& var);

	/// <summary>
	/// Collect data statistics on Opencl device
	/// </summary>
	/// <param name="m_ocl_config">- Opencl configuration structure</param>
	/// <param name="configuration">- Program configuration structure</param>
	/// <param name="data">- Data pointer</param>
	/// <param name="data_count">- Data count to process</param>
	/// <returns>Local data statistics</returns>
	SDataStat run_statistics_on_GPU(SOpenCLConfig& m_ocl_config, SConfig& configuration, double* data, long long begin, long long end);

	/// <summary>
	/// Init opencl 
	/// </summary>
	/// <param name="configuration">- Program configuration structure</param>
	/// <param name="file">- kernel file</param>
	/// <param name="kernel_name">- kernel name</param>
	/// <returns>Opencl configuration structure</returns>
	SOpenCLConfig init(SConfig& configuration, const std::string& file, const char* kernel_name);

	/// <summary>
	/// Update OpenCL kernel program
	/// </summary>
	/// <param name="configuration">- Program configuration structure</param>
	/// <param name="file">- kernel file</param>
	/// <param name="kernel_name">- kernel name</param>
	void set_kernel_program(SOpenCLConfig& opencl, const std::string& file, const char* kernel_name);

	/// <summary>
	/// Create OpenCl kernel with init program
	/// </summary>
	/// <param name="opencl">- Opencl configuration structure</param>
	/// <param name="kernel_name">- kernel name</param>
	void create_kernel(SOpenCLConfig& opencl, const char* kernel_name);

	/// <summary>
	/// Create OpenCl kernel with program
	/// </summary>
	/// <param name="opencl">- Opencl configuration structure</param>
	/// <param name="file">- kernel file</param>
	void create_kernel_program(SOpenCLConfig& opencl, const std::string& file);

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