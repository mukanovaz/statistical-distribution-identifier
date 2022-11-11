#include "smp_solver.h"
#include "../rss/statistics.cpp"
#include "../file_mapping.h"
#include "../histogram/histogram.cpp"

#undef min
#undef max

#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

namespace ppr::parallel
{
	SResult run(SConfig& configuration)
	{
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		const double* data = mapping.GetData();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}

		//  ================ [Get statistics]
		RunningStatParallel stat(data);
		tbb::parallel_reduce(tbb::blocked_range<std::size_t>(1, mapping.GetCount()), stat);
		
		//  ================ [Fit params]
		// Gauss maximum likelihood estimators
		double gauss_mean = stat.Mean();
		double gauss_variance = stat.Variance();
		double gauss_sd = stat.StandardDeviation();

		// Exponential maximum likelihood estimators
		double exp_lambda = static_cast<double>(stat.NumDataValues()) / stat.Sum();

		// Poisson likelihood estimators
		double poisson_lambda = stat.Mean();

		// Uniform likelihood estimators
		double a = stat.Get_Min();
		double b = stat.Get_Max();

		//  ================ [Create histogram]

		mapping.UnmapFile();

		//  ================ [Get propability density of histogram]

		// ================ [Calculate RSS]
		return SResult::error_res(EExitStatus::STAT);
	}

}