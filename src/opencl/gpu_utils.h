#pragma once
#include <CL/cl.hpp>
#include <string>
#include "../config.h"

namespace ppr::gpu
{
	void RunStatisticsOnGPU(SOpenCLConfig& opencl, SConfig& configuration, double* data,
		std::vector<double>& out_sum,
		std::vector<double>& out_min,
		std::vector<double>& out_max);

	SOpenCLConfig Init(SConfig& configuration, const std::string& file, const char* kernel_name);
	void UpdateProgram(SOpenCLConfig& opencl, const std::string& file, const char* kernel_name);
	void CreateKernel(SOpenCLConfig& opencl, const char* kernel_name);
	void CreateProgram(SOpenCLConfig& opencl, const std::string& file);
	void FindDevices(std::vector<cl::Platform>& platforms, std::vector<cl::Device>& all_devices, std::vector<std::string>& user_devices);

	std::string GetCLErrorString(cl_int error);
}