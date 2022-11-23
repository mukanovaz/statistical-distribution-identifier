#pragma once
#include <CL/cl.hpp>
#include <string>

namespace ppr::gpu
{
	cl::Program CreateProgram(const cl::Device* devices, const char* source);
	void FindDevices(std::vector<cl::Platform>& platforms, std::vector<cl::Device>& all_devices, std::vector<std::string>& user_devices);
	std::string GetCLErrorString(cl_int error);
}