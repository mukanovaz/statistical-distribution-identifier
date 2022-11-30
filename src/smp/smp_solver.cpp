#include "smp_solver.h"
#include <vector>
#include <future>

namespace ppr::parallel
{
	SResult run(SConfig& configuration)
	{
		//  ================ [Init TBB]
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		//  ================ [Allocations]
		SOpenCLConfig opencl; // Not using here
		SHistogram hist;
		SResult res;
		SDataStat stat;
		unsigned int data_count = mapping.GetCount();
		size_t chunks_count = mapping.GetFileLen() / mapping.GetGranularity();
		std::vector<std::future<SDataStat>> workersStat((configuration.thread_count * chunks_count) + 1);

		//  ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		mapping.ReadInChunksStat(configuration, opencl, stat);
		tbb::tick_count t1 = tbb::tick_count::now();
		std::cout << "Statistics:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		
		// Find mean
		stat.mean = stat.sum / stat.n;

		//  ================ [Create frequency histogram]
		// Find histogram limits
		hist.binCount = log2(stat.n) + 1;
		hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		std::vector<int> histogramFreq(static_cast<int>(hist.binCount + 1));
		std::vector<double> histogramDensity(static_cast<int>(hist.binCount));
		cl_int err = 0;

		// Run
		t0 = tbb::tick_count::now();
		mapping.ReadInChunksHist(hist, configuration, opencl, stat, histogramFreq);
		t1 = tbb::tick_count::now();
		std::cout << "Histogram:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		// Find variance
		stat.variance = stat.variance / stat.n;

		//  ================ [Create density histogram]
		ppr::executor::ComputePropabilityDensityOfHistogram(hist, histogramFreq, histogramDensity, stat.n);

		//  ================ [Fit params]
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
		ppr::executor::CalculateHistogramRSSOnCPU(res, arena, histogramDensity, hist);

		//	================ [Analyze]
		ppr::executor::AnalyzeResults(res);

		std::cout << "Finish." << std::endl;
		return res;
	}

	void GetStatisticsCPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		// Find rest of a statistics on CPU
		RunningStatParallel stat_cpu(data, opencl.data_count_for_cpu);
		ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat_cpu, opencl.data_count_for_cpu + 1, data_count);

		// Agregate results results
		stat.n += stat_cpu.NumDataValues();
		stat.min = std::min({ stat.min, std::min({ stat.min, stat_cpu.Get_Min() }) });
		stat.max = std::max({ stat.max, std::max({ stat.max, stat_cpu.Get_Max() }) });
		stat.sum += stat_cpu.Sum();
		stat.isNegative = stat.isNegative || stat_cpu.IsNegative();
	}

	void CreateFrequencyHistogramCPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		// Run on CPU
		ppr::hist::HistogramParallel hist_cpu(hist.binCount, hist.binSize, stat.min, stat.max, data, stat.mean);
		ppr::executor::RunOnCPU<ppr::hist::HistogramParallel>(arena, hist_cpu, opencl.data_count_for_cpu, data_count);

		// Transform vector
		std::transform(histogram.begin(), histogram.end(), hist_cpu.m_bucketFrequency.begin(), histogram.begin(), std::plus<int>());

		stat.variance += hist_cpu.m_var; // GPU + CPU "half" variance
	}


}