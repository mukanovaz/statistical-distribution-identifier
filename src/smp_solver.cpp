#include "include/smp_solver.h"
#include "include/watchdog.h"
#include <vector>
#include <future>

namespace ppr::parallel
{
	SResult run(SConfig& configuration)
	{
		tbb::tick_count total1;
		total1 = tbb::tick_count::now();
		//  ================ [Init TBB]
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Map input file]
		File_mapping mapping(configuration);

		//  ================ [Allocations]
		tbb::tick_count total2;
		tbb::tick_count t0;
		tbb::tick_count t1;
		int stage = 0;
		SOpenCLConfig opencl;
		SHistogram hist;
		SResult res;
		SDataStat stat;
		std::vector<int> tmp(0);
		std::vector<int> histogramFreq(0);			// Will resize after collecting statistics
		std::vector<double> histogramDensity(0);	// Will resize after collecting statistics
		unsigned int data_count = mapping.get_count();

		//  ================ [Start Watchdog]
		ppr::watchdog::start_watchdog(configuration, stat, hist, stage, histogramFreq, histogramDensity, data_count);
		
		//  ================ [Get statistics]
		if (configuration.use_optimalization)
		{
			// Optimized run
			t0 = tbb::tick_count::now();
			mapping.read_in_chunks_cpu(hist, configuration, opencl, stat, EIteration::STAT, tmp);
			t1 = tbb::tick_count::now();
		}
		else
		{
			// TBB run
			t0 = tbb::tick_count::now();
			mapping.read_in_chunks(hist, configuration, opencl, stat, arena, tmp, &get_statistics_CPU);
			t1 = tbb::tick_count::now();
		}

		res.total_stat_time = (t1 - t0).seconds();

		// Find mean
		stat.mean = stat.sum / stat.n;

		//  ================ [Create frequency histogram]
		// Find histogram limits
		hist.binCount = log2(stat.n) + 1;
		hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		histogramFreq.resize(static_cast<int>(hist.binCount + 1));
		histogramDensity.resize(static_cast<int>(hist.binCount + 1));

		stage = 1;
		// Run
		if (configuration.use_optimalization)
		{
			// Optimized run
			t0 = tbb::tick_count::now();
			mapping.read_in_chunks_cpu(hist, configuration, opencl, stat, EIteration::HIST, histogramFreq);
			t1 = tbb::tick_count::now();
		}
		else
		{
			// TBB run
			t0 = tbb::tick_count::now();
			mapping.read_in_chunks(hist, configuration, opencl, stat, arena, histogramFreq, &create_frequency_histogram_CPU);
			t1 = tbb::tick_count::now();
		}
		res.total_hist_time = (t1 - t0).seconds();

		// Find variance
		stat.variance = stat.variance / stat.n;

		//  ================ [Create density histogram]
		stage = 2;
		ppr::executor::compute_propability_density_histogram(hist, histogramFreq, histogramDensity, stat.n);

		//  ================ [Fit params using Maximum likelihood estimation]
		res.isNegative = stat.min < 0;
		res.isInteger = std::floor(stat.sum) == stat.sum;

		// Gauss maximum likelihood estimators
		res.gauss_mean = stat.mean;
		res.gauss_variance = stat.variance;
		res.gauss_stdev = sqrt(stat.variance);

		// Exponential maximum likelihood estimators
		res.exp_lambda = stat.n / stat.sum;

		// Poisson likelihood estimators
		res.poisson_lambda = stat.sum / stat.n;

		// Uniform likelihood estimators
		res.uniform_a = stat.min;
		res.uniform_b = stat.max;

		//	================ [Calculate RSS]
		stage = 3;
		ppr::executor::calculate_histogram_RSS_with_tbb(res, arena, histogramDensity, hist);

		//	================ [Analyze Results]
		ppr::executor::analyze_results(res);

		total2 = tbb::tick_count::now();
		
		res.total_time = (total2 - total1).seconds();
		stage = 4;

		std::cout << "\t\t\t[Statistics]" << std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;
		std::cout << "> n:\t\t\t" << stat.n << std::endl;
		std::cout << "> sum:\t\t\t" << stat.sum << std::endl;
		std::cout << "> mean:\t\t\t" << stat.mean << std::endl;
		std::cout << "> variance:\t\t" << stat.variance << std::endl;
		std::cout << std::endl;

		return res;
	}

	void get_statistics_CPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		// Find rest of a statistics on CPU
		RunningStatParallel stat_cpu(data, opencl.data_count_for_cpu);
		ppr::executor::run_with_tbb<RunningStatParallel>(arena, stat_cpu, opencl.data_count_for_cpu + 1, data_count);

		// Agregate results results
		stat.n += stat_cpu.NumDataValues();
		stat.min = std::min({ stat.min, std::min({ stat.min, stat_cpu.Get_Min() }) });
		stat.max = std::max({ stat.max, std::max({ stat.max, stat_cpu.Get_Max() }) });
		stat.sum += stat_cpu.Sum();
		stat.isNegative = stat.isNegative || stat_cpu.IsNegative();
	}

	void create_frequency_histogram_CPU(SHistogram& hist, SConfig&, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		// Run on CPU
		ppr::hist::Histogram_parallel hist_cpu(hist.binCount, hist.binSize, stat.min, stat.max, data, stat.mean);
		ppr::executor::run_with_tbb<ppr::hist::Histogram_parallel>(arena, hist_cpu, opencl.data_count_for_cpu, data_count);

		// Transform vector
		std::transform(histogram.begin(), histogram.end(), hist_cpu.m_bucketFrequency.begin(), histogram.begin(), std::plus<int>());

		stat.variance += hist_cpu.m_var;
	}


}