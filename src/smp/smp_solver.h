#pragma once
#include "../data.h"
#include "../config.h"
#include "../file_mapping.h"
#include "../executor.h"

namespace ppr::parallel
{
	SResult run(SConfig& configuration);
	SResult run(SConfig& configuration);
	void GetStatisticsCPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram);
	void CreateFrequencyHistogramCPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram);
}
