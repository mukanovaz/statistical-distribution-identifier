#pragma once
#include "../data.h"
#include "../config.h"
#include "gpu_utils.h" 
#include "../file_mapping.h"
#include "../executor.h"

namespace ppr::gpu
{
	SResult run(SConfig& configuration);
	void GetStatistics(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram);
	void CreateFrequencyHistogram(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram);
}
