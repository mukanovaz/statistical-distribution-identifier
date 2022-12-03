#include "gpu_solver.h"
#include "../executor.h"
#include "../watchdog.h"

namespace ppr::gpu
{
	SResult run(SConfig& configuration)
	{
		tbb::tick_count total1;
		total1 = tbb::tick_count::now();
		//  ================ [Init TBB]
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Init OpenCL]
		SOpenCLConfig opencl = ppr::gpu::Init(configuration, STAT_KERNEL, STAT_KERNEL_NAME);

		//  ================ [Get file mapping]
		FileMapping mapping(configuration);

		//  ================ [Allocations]
		tbb::tick_count total2;
		int stage = 0;
		SHistogram hist;
		SResult res;
		SDataStat stat;
		std::vector<int> tmp(0);
		std::vector<int> histogramFreq(0);
		std::vector<double> histogramDensity(0);
		unsigned int data_count = mapping.GetCount();

		//  ================ [Start Watchdog]
		ppr::watchdog::start_watchdog(stat, hist, stage, histogramFreq, histogramDensity, data_count);

		//  ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		mapping.ReadInChunks(hist, configuration, opencl, stat, arena, tmp, &GetStatistics);
		tbb::tick_count t1 = tbb::tick_count::now();
		
		res.total_stat_time = (t1 - t0).seconds();

		// Find mean
		stat.mean = stat.sum / stat.n;

		//  ================ [Create frequency histogram]
		// Update kernel program
		ppr::gpu::UpdateProgram(opencl, HIST_KERNEL, HIST_KERNEL_NAME);

		// Find histogram limits
		hist.binCount = log2(stat.n) + 1;
		hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		histogramFreq.resize(static_cast<int>(hist.binCount + 1));
		histogramDensity.resize(static_cast<int>(hist.binCount + 1));
		cl_int err = 0;

		// Run
		stage = 1;
		t0 = tbb::tick_count::now();
		mapping.ReadInChunks(hist, configuration, opencl, stat, arena, histogramFreq, &CreateFrequencyHistogram);
		t1 = tbb::tick_count::now();
		
		res.total_hist_time = (t1 - t0).seconds();

		// Find variance
		stat.variance = stat.variance / stat.n;

		//  ================ [Create density histogram]
		stage = 2;
		ppr::executor::ComputePropabilityDensityOfHistogram(hist, histogramFreq, histogramDensity, stat.n);

		//	================ [Fit params using Maximum likelihood estimation]
		res.isNegative = stat.isNegative;
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
		ppr::executor::CalculateHistogramRSSOnCPU(res, arena, histogramDensity, hist);

		//	================ [Analyze Results]
		ppr::executor::AnalyzeResults(res);

		total2 = tbb::tick_count::now();

		res.total_time = (total2 - total1).seconds();
		stage = 4;

		return res;
	}

	void GetStatistics(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		if (opencl.data_count_for_cpu < data_count)
		{
			// Find rest of a statistics on CPU
			RunningStatParallel stat_cpu(data, opencl.data_count_for_cpu);
			ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat_cpu, opencl.data_count_for_cpu + 1, data_count);

			// Find statistics on GPU
			SDataStat stat_gpu = ppr::executor::RunStatisticsOnGPU(opencl, configuration, data);

			// Agregate results results
			stat.n += stat_gpu.n + stat_cpu.NumDataValues();
			stat.min = std::min({ stat.min, std::min({ stat_gpu.min, stat_cpu.Get_Min() }) });
			stat.max = std::max({ stat.max, std::max({ stat_gpu.max, stat_cpu.Get_Max() }) });
			stat.sum += stat_gpu.sum + stat_cpu.Sum();
			stat.isNegative = stat.isNegative || stat_gpu.isNegative || stat_cpu.IsNegative();
		}
		else
		{
			// Find statistics on GPU
			SDataStat stat_gpu = ppr::executor::RunStatisticsOnGPU(opencl, configuration, data);

			// Agregate results results
			stat.n += stat_gpu.n;
			stat.min = std::min({ stat_gpu.min, stat.min});
			stat.max = std::max({ stat_gpu.max, stat.max });
			stat.sum += stat_gpu.sum;
			stat.isNegative = stat.isNegative || stat_gpu.isNegative;
		}
	}

	void CreateFrequencyHistogram(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		if (opencl.data_count_for_cpu < data_count)
		{
			// Run on CPU
			ppr::hist::HistogramParallel hist_cpu(hist.binCount, hist.binSize, stat.min, stat.max, data, stat.mean);
			ppr::executor::RunOnCPU<ppr::hist::HistogramParallel>(arena, hist_cpu, opencl.data_count_for_cpu, data_count);

			// Transform vector
			std::transform(histogram.begin(), histogram.end(), hist_cpu.m_bucketFrequency.begin(), histogram.begin(), std::plus<int>());
			
			stat.variance += hist_cpu.m_var; // GPU + CPU "half" variance
		}

		// Run on GPU
		ppr::executor::RunHistogramOnGPU(opencl, stat, hist, data, histogram);
	}
}
