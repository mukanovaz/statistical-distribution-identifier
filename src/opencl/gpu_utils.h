#pragma once
#include <CL/cl.hpp>
#include <string>
#include "../config.h"

namespace ppr::gpu
{
	cl::Program CreateProgram(const cl::Device* devices, const std::string& file);
	SOpenCLConfig Init(SConfig& configuration, const std::string& file);
	void FindDevices(std::vector<cl::Platform>& platforms, std::vector<cl::Device>& all_devices, std::vector<std::string>& user_devices);
	std::string GetCLErrorString(cl_int error);

	//constexpr const char* statisctic_kernel = R"awdwad";
}