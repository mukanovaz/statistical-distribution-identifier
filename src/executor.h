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

	double SumVectorOnCPU(tbb::task_arena& arena, std::vector<double> data);

	SDataStat RunStatisticsOnGPU(SOpenCLConfig& opencl, SConfig& configuration, tbb::task_arena& arena, double* data);

	void RunHistogramOnGPU(SOpenCLConfig& opencl, SDataStat& data_stat, SHistogram& histogram, tbb::task_arena& arena, double* data, std::vector<int>& freq_buckets);
}
