#pragma once
#include "../data.h"
#include "../config.h"
#include "gpu_utils.h" 
#include "../file_mapping.h"
#include "../executor.h"

namespace ppr::gpu
{
	SResult run(SConfig& configuration);
	SResult run2(SConfig& configuration);
	void GetStatistics(SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<double>& histogram);
	void CreateFrequencyHistogram(SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<double>& histogram);
	void CalculateHistogramRSS(SResult& res, tbb::task_arena& arena, std::vector<double>& histogramDensity, SHistogram& hist);
	void AnalyzeResults(SResult& res);
}
