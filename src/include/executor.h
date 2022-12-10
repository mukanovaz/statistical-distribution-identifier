#pragma once
#include "statistics.cpp"
#include "histogram.h"
#include "gpu_utils.h"
#include <numeric>
#include <vector>
#include <algorithm>
#include <functional>
#include <array>

#undef min
#undef max

#include <tbb/tick_count.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/blocked_range.h>


namespace ppr::executor
{
	/// <summary>
	/// Run TBB class using tbb::parallel_reduce algorithm
	/// </summary>
	/// <typeparam name="T">- TBB class name</typeparam>
	/// <param name="arena">- TBB arena</param>
	/// <param name="class_to_execute">- Class to execute</param>
	/// <param name="begin">- Begin interval</param>
	/// <param name="end">- End interval</param>
	/// <returns>Processed time</returns>
	template <typename T>
	double run_with_tbb(tbb::task_arena& arena, T& class_to_execute, unsigned long long begin, unsigned long long end)
	{
		tbb::tick_count t0 = tbb::tick_count::now();
		arena.execute([&]() {
			tbb::parallel_reduce(tbb::blocked_range<std::size_t>(begin, end), class_to_execute);
			});
		tbb::tick_count t1 = tbb::tick_count::now();

		return (t1 - t0).seconds();
	}

	/// <summary>
	/// Sum all vector elements using TBB algorithm
	/// </summary>
	/// <param name="arena">- TBB arena</param>
	/// <param name="data">- vector with data</param>
	/// <returns>Sum of all elements</returns>
	double sum_vector_tbb(tbb::task_arena& arena, std::vector<double> data);

	/// <summary>
	/// Analyze RSS results
	/// </summary>
	/// <param name="res">Results</param>
	void analyze_results(SResult& res);
	
	/// <summary>
	/// Transform frequency histogram to propability density histogram
	/// </summary>
	/// <param name="hist">- Histogram configuration structure</param>
	/// <param name="bucket_frequency">- Frequency histogram reference</param>
	/// <param name="bucket_density">- Density histogram reference</param>
	/// <param name="count">- All data count</param>
	void compute_propability_density_histogram(SHistogram& hist, std::vector<int>& bucket_frequency, 
		std::vector<double>& bucket_density, unsigned long long data_count);
}
