#include "gpu_solver.h"
#include "gpu_utils.h"
#include "../file_mapping.h"
#include "../executor.h"


namespace ppr::gpu
{
	SResult run(SConfig& configuration)
	{
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Init OpenCL]
		// Find CL devices
		
		SOpenCLConfig opencl = ppr::gpu::Init(configuration, "D:/Study/ZCU/5.semestr/PPR/kiv-ppr/msvc/statistics_kernel.cl");		

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		double* data = mapping.GetData();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}
		

		//  ================ [Get statistics]

		// Get number of data, which we want to process on GPU
		auto wg_count = mapping.GetCount() / opencl.wg_size;
		auto data_count_for_gpu = mapping.GetCount() - (mapping.GetCount() % opencl.wg_size);

		// The rest of the data we will process on CPU
		unsigned long data_count_for_cpu = data_count_for_gpu + 1;

		// Find rest of a statistics on CPU
		RunningStatParallel stat(data, data_count_for_cpu);
		ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat, 1, mapping.GetCount());

		// Find statistics on GPU
		SDataStat stat_gpu = ppr::executor::RunStatisticsOnGPU(opencl, configuration, arena, data_count_for_gpu, wg_count, data);

		

		//  ================ [Fit params]

		//  ================ [Create frequency histogram]

		//================ [Unmap file]
		mapping.UnmapFile();

		//  ================ [Get propability density of histogram]

		// ================ [Calculate RSS]
		//
		return SResult::error_res(EExitStatus::STAT);
	}
}
