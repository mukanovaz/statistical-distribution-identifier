#pragma once
#include <CL/cl.hpp>
#include <string>
#include "../config.h"
#include "../data.h"

namespace ppr::gpu
{
	void RunHistogramOnGPU(SOpenCLConfig& opencl, SConfig& configuration, SHistogram& hist, SDataStat& data_stat,
		double* data, int data_count, std::vector<int>& freq_buckets, double& var);
	SDataStat RunStatisticsOnGPU(SOpenCLConfig& m_ocl_config, SConfig& configuration, double* data, int data_count);

	SOpenCLConfig Init(SConfig& configuration, const std::string& file, const char* kernel_name);
	void UpdateProgram(SOpenCLConfig& opencl, const std::string& file, const char* kernel_name);
	void CreateKernel(SOpenCLConfig& opencl, const char* kernel_name);
	void CreateProgram(SOpenCLConfig& opencl, const std::string& file);
	void FindDevices(std::vector<cl::Platform>& platforms, std::vector<cl::Device>& all_devices, std::vector<std::string>& user_devices);

	std::string GetCLErrorString(cl_int error);
}