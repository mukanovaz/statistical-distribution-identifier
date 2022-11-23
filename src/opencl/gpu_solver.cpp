#include "gpu_solver.h"
#include "../file_mapping.h"
#include "gpu_utils.h"


namespace ppr::gpu
{
	SResult run(SConfig& configuration)
	{
		//  ================ [Init OpenCL]
		// Find CL devices
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		std::vector<cl::Device> devices(configuration.cl_devices_name.size());
		ppr::gpu::FindDevices(platforms, devices, configuration.cl_devices_name);
		
		cl::Program program = ppr::gpu::CreateProgram(&devices.front(), "");

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		const double* data = mapping.GetData();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}

		//  ================ [Get statistics]

		//  ================ [Fit params]

		//  ================ [Create frequency histogram]

		//================ [Unmap file]
		mapping.UnmapFile();

		//  ================ [Get propability density of histogram]

		// ================ [Calculate RSS]

		return SResult::error_res(EExitStatus::STAT);
	}
}
