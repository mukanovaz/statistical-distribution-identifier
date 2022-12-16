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
		ppr::gpu::SOpenCLConfig opencl;
		SHistogram hist;
		SResult res;
		SDataStat stat;
		std::vector<int> tmp(0);
		std::vector<int> histogramFreq(0);			// Will resize after collecting statistics
		std::vector<double> histogramDensity(0);	// Will resize after collecting statistics
		long data_count = mapping.get_count();

		//  ================ [Start Watchdog]
		std::thread watchdog = ppr::watchdog::start_watchdog(configuration, stat, hist, stage, histogramFreq, histogramDensity, data_count);
		
		//  ================ [Get statistics]
		if (configuration.use_optimalization)
		{
			// Optimized run
			t0 = tbb::tick_count::now();
			mapping.read_in_one_chunk_cpu(hist, configuration, opencl, stat, EIteration::STAT, tmp);
			t1 = tbb::tick_count::now();
		}
		else
		{
			// TBB run
			t0 = tbb::tick_count::now();
			mapping.read_in_chunks_tbb(hist, configuration, opencl, stat, arena, tmp, &get_statistics_CPU);
			t1 = tbb::tick_count::now();
		}

		res.total_stat_time = (t1 - t0).seconds();

		//  ================ [Fit params using Maximum likelihood estimation]

		res.isNegative = stat.min < 0;
		res.isInteger = std::floor(stat.sum) == stat.sum;

		// Find mean
		stat.mean = stat.sum / stat.n;
		
		// Poisson likelihood estimators
		res.poisson_lambda = stat.sum / stat.n;

		//  ================ [Create frequency histogram]
		
		// Find histogram limits
		double bin_count = 0.0;
		double bin_size = 0.0;

		// If data can belongs to poisson distribution, we should use integer intervals
		if (!res.isNegative && res.isInteger && res.poisson_lambda > 0)
		{
			hist.binCount = static_cast<int>(stat.max - stat.min);
			hist.binSize = 1.0;
		}
		else
		{
			hist.binCount = static_cast<int>(log2(stat.n)) + 2;
			hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		}
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		histogramFreq.resize(static_cast<int>(hist.binCount));
		histogramDensity.resize(static_cast<int>(hist.binCount));

		stage = 1;

		// Run
		if (configuration.use_optimalization)
		{
			// Optimized run
			t0 = tbb::tick_count::now();
			mapping.read_in_one_chunk_cpu(hist, configuration, opencl, stat, EIteration::HIST, histogramFreq);
			t1 = tbb::tick_count::now();
		}
		else
		{
			// TBB run
			t0 = tbb::tick_count::now();
			mapping.read_in_chunks_tbb(hist, configuration, opencl, stat, arena, histogramFreq, &create_frequency_histogram_CPU);
			t1 = tbb::tick_count::now();
		}
		res.total_hist_time = (t1 - t0).seconds();


		//  ================ [Fit params using Maximum likelihood estimation]

		// Find variance
		stat.variance = stat.variance / stat.n;

		// Gauss maximum likelihood estimators
		res.gauss_mean = stat.mean;
		res.gauss_variance = stat.variance;
		res.gauss_stdev = sqrt(stat.variance);

		// Exponential maximum likelihood estimators
		res.exp_lambda = stat.n / stat.sum;

		// Uniform likelihood estimators
		res.uniform_a = stat.min;
		res.uniform_b = stat.max;


		//  ================ [Create density histogram]
		stage = 2;
		ppr::executor::compute_propability_density_histogram(hist, histogramFreq, histogramDensity, stat.n);

		//	================ [Calculate RSS]
		stage = 3;
		ppr::parallel::calculate_histogram_RSS_cpu(res, histogramDensity, hist);

		//	================ [Analyze Results]
		ppr::executor::analyze_results(res);

		total2 = tbb::tick_count::now();
		
		res.total_time = (total2 - total1).seconds();
		stage = 4;

		print_stat(stat, res);

		// Wait until watchdog will finish
		watchdog.join();
		return res;
	}

	void get_statistics_CPU(SHistogram& hist, SConfig& configuration, ppr::gpu::SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned long long data_count, double* data, std::vector<int>& histogram)
	{
		// Find rest of a statistics on CPU
		Running_stat_parallel stat_cpu(data, opencl.data_count_for_cpu);
		ppr::executor::run_with_tbb<Running_stat_parallel>(arena, stat_cpu, opencl.data_count_for_cpu + 1, data_count);

		// Agregate results results
		stat.n += stat_cpu.NumDataValues();
		stat.min = std::min({ stat.min, std::min({ stat.min, stat_cpu.Get_Min() }) });
		stat.max = std::max({ stat.max, std::max({ stat.max, stat_cpu.Get_Max() }) });
		stat.sum += stat_cpu.Sum();
		stat.isNegative = stat.isNegative || stat_cpu.IsNegative();
	}

	void create_frequency_histogram_CPU(SHistogram& hist, SConfig&, ppr::gpu::SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned long long data_count, double* data, std::vector<int>& histogram)
	{
		// Run on CPU
		ppr::hist::Histogram_parallel hist_cpu(hist.binCount, hist.binSize, stat.min, stat.max, data, stat.mean);
		ppr::executor::run_with_tbb<ppr::hist::Histogram_parallel>(arena, hist_cpu, opencl.data_count_for_cpu, data_count);

		// Transform vector
		std::transform(histogram.begin(), histogram.end(), hist_cpu.m_bucketFrequency.begin(), histogram.begin(), std::plus<int>());

		stat.variance += hist_cpu.m_var;
	}


}