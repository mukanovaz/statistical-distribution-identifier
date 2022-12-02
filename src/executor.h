#pragma once
#undef min
#undef max

#include <tbb/tick_count.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/blocked_range.h>
#include "rss/statistics.cpp"
#include "histogram/histogram.cpp"
#include "opencl/gpu_utils.h"
#include <numeric>
#include <vector>
#include <algorithm>
#include <functional>
#include <array>

namespace ppr::executor
{
	template <typename T>
	double RunOnCPU(tbb::task_arena& arena, T& class_to_execute, int begin, int end)
	{
		tbb::tick_count t0 = tbb::tick_count::now();
		arena.execute([&]() {
			tbb::parallel_reduce(tbb::blocked_range<std::size_t>(begin, end), class_to_execute);
			});
		tbb::tick_count t1 = tbb::tick_count::now();

		return (t1 - t0).seconds();
	}

	void CalculateHistogramRSSOnCPU(SResult& res, tbb::task_arena& arena, std::vector<double>& histogramDensity, SHistogram& hist);

	double SumVectorOnCPU(tbb::task_arena& arena, std::vector<double> data);

	SDataStat RunStatisticsOnGPU(SOpenCLConfig& opencl, SConfig& configuration, double* data);

	void RunHistogramOnGPU(SOpenCLConfig& opencl, SDataStat& data_stat, SHistogram& histogram, double* data, std::vector<int>& freq_buckets);

	void AnalyzeResults(SResult& res);

	void ComputePropabilityDensityOfHistogram(SHistogram& hist, std::vector<int>& bucket_frequency, std::vector<double>& bucket_density, double count);
}
