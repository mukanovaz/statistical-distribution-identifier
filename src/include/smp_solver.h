#pragma once
#include "data.h"
#include "config.h"
#include "file_mapping.h"
#include "executor.h"

#include "smp_utils.h"

namespace ppr::parallel
{
	/// <summary>
	/// Starting function which runs on CPU only
	/// </summary>
	/// <param name="configuration">Program configuration structure</param>
	/// <returns>Computing results</returns>
	SResult run(SConfig& configuration);

	/// <summary>
	/// Calculate data statistics using Intel TBB algorithms. Calls from file_mapping.h > read_in_chunks()
	/// </summary>
	/// <param name="hist">Histogram configuration structure</param>
	/// <param name="configuration">Program configuration structure</param>
	/// <param name="opencl">Opencl configuration structure</param>
	/// <param name="stat">Statistics structure</param>
	/// <param name="arena">TBB arena</param>
	/// <param name="data_count">Data count for processing</param>
	/// <param name="data">Data pointer</param>
	/// <param name="histogram">Frequency histogram reference</param>
	void get_statistics_CPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, 
		tbb::task_arena& arena, unsigned long long data_count, double* data, std::vector<int>& histogram);

	/// <summary>
	/// Create frequency histogram using Intel TBB algorithms. Calls from file_mapping.h > read_in_chunks()
	/// </summary>
	/// <param name="hist">Histogram configuration structure</param>
	/// <param name="opencl">Opencl configuration structure</param>
	/// <param name="stat">Statistics structure</param>
	/// <param name="arena">TBB arena</param>
	/// <param name="data_count">Data count for processing</param>
	/// <param name="data">Data pointer</param>
	/// <param name="histogram">Frequency histogram reference</param>
	void create_frequency_histogram_CPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl,
		SDataStat& stat, tbb::task_arena& arena, unsigned long long data_count, double* data, std::vector<int>& histogram);
}
