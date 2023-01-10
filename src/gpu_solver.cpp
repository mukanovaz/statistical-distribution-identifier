#include "include/gpu_solver.h"
#include "include/executor.h"
#include "include/watchdog.h"
#include "include/histogram.h"

namespace ppr::gpu
{
	SResult run(SConfig& configuration)
	{
		tbb::tick_count total1;
		total1 = tbb::tick_count::now();
		//  ================ [Init TBB]
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Init OpenCL]
		/*ppr::gpu::SOpenCLConfig opencl = ppr::gpu::init(configuration, STAT_KERNEL, STAT_KERNEL_NAME);*/


		//  ================ [Get file mapping]
		File_mapping mapping(configuration);

		//  ================ [Allocations]
		tbb::tick_count total2;
		int stage = 0;
		SHistogram hist;
		SResult res;
		SDataStat stat;
		std::vector<int> tmp(0);
		std::vector<int> histogramFreq(0);
		std::vector<double> histogramDensity(0);
		long data_count = mapping.get_count();
		double* data = mapping.get_data();

		//  ================ [Start Watchdog]
		std::thread watchdog = ppr::watchdog::start_watchdog(configuration, stat, hist, stage, histogramFreq, histogramDensity, data_count);

		//  ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		mapping.read_in_chunks_gpu(hist, configuration, stat, EIteration::STAT, histogramFreq);
		tbb::tick_count t1 = tbb::tick_count::now();
		
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
			hist.binSize = (stat.max - stat.min) / (static_cast<double>(hist.binCount) - 1.0);
		}
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		histogramFreq.resize(static_cast<int>(hist.binCount));
		histogramDensity.resize(static_cast<int>(hist.binCount));
		cl_int err = 0;

		// Run
		stage = 1;
		t0 = tbb::tick_count::now();
		mapping.read_in_chunks_gpu(hist, configuration, stat, EIteration::HIST, histogramFreq);
		t1 = tbb::tick_count::now();
		
		res.total_hist_time = (t1 - t0).seconds();

		//  ================ [Create density histogram]
		stage = 2;
		ppr::executor::compute_propability_density_histogram(hist, histogramFreq, histogramDensity, stat.n);

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

		//	================ [Calculate RSS]
		stage = 3;
		ppr::parallel::calculate_histogram_RSS_cpu(res, histogramDensity, hist);

		//	================ [Analyze Results]
		ppr::executor::analyze_results(res);

		total2 = tbb::tick_count::now();

		res.total_time = (total2 - total1).seconds();
		stage = 4;

		std::cout << "\t\t\t[Statistics]" << std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;
		std::cout << "> n:\t\t\t\t" << stat.n << std::endl;
		std::cout << "> sum:\t\t\t\t" << stat.sum << std::endl;
		std::cout << "> mean:\t\t\t\t" << stat.mean << std::endl;
		std::cout << "> variance:\t\t\t" << stat.variance << std::endl;
		std::cout << "> min:\t\t\t\t" << stat.min << std::endl;
		std::cout << "> max:\t\t\t\t" << stat.max << std::endl;
		std::cout << "> isNegative:\t\t\t" << res.isNegative << std::endl;
		std::cout << "> isInteger:\t\t\t" << res.isInteger << std::endl;

		watchdog.join();
		return res;
	}

}
