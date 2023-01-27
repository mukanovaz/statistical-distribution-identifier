#pragma once
#include "data.h"
#include "config.h"
#include "executor.h"

#include "smp_utils.h"

namespace ppr::solver
{
	SResult run(SConfig& configuration);

	void compute_histogram(std::vector<cl::Device> devices, SHistogram& hist, SConfig& configuration, SDataStat& stat, const unsigned long long file_len, DWORD64 data_count,
		std::vector<int>& histogram);

	void compute_statistics(std::vector<cl::Device> devices, SConfig& configuration, SDataStat& stat, const unsigned long long file_len, DWORD64 data_count);

}