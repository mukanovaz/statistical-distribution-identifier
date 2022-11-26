#include "gpu_solver.h"
#include <algorithm>
#include <functional>
#include <array>

namespace ppr::gpu
{
	SResult run(SConfig& configuration)
	{
		//  ================ [Init TBB]
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Init OpenCL]
		SOpenCLConfig opencl = ppr::gpu::Init(configuration, STAT_KERNEL, STAT_KERNEL_NAME);

		//  ================ [Get file info]
		FileMapping mapping(configuration.input_fn);

		//  ================ [Allocations]
		std::vector<double> tmp(0);
		unsigned int data_count = mapping.GetCount();
		SDataStat stat;

		//  ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		mapping.ReadInChunks(L"D:/Study/ZCU/5.semestr/PPR/rozdeleni/gauss_big", configuration, opencl, stat, arena, tmp, &GetStatistics);
		tbb::tick_count t1 = tbb::tick_count::now();
		std::cout << (t1 - t0).seconds() << "sec." << std::endl;

		//  ================ [Create frequency histogram]
		//mapping.ReadInChunks(L"D:/Study/ZCU/5.semestr/PPR/rozdeleni/gauss", configuration, opencl, stat, arena, tmp, &CreateFrequencyHistogram);

		return SResult::error_res(EExitStatus::STAT);
	}

	void GetStatistics(SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<double>& histogram)
	{
		
		if (opencl.data_count_for_cpu < data_count)
		{
			// Find rest of a statistics on CPU
			RunningStatParallel stat_cpu(data, opencl.data_count_for_cpu);
			ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat_cpu, opencl.data_count_for_cpu, data_count);

			// Find statistics on GPU
			SDataStat stat_gpu = ppr::executor::RunStatisticsOnGPU(opencl, configuration, arena, data);

			// Agregate results results
			stat.n += stat_gpu.n + stat_cpu.NumDataValues();
			stat.min = std::min({ stat_gpu.min, stat_cpu.Get_Min() });
			stat.max = std::max({ stat_gpu.max, stat_cpu.Get_Max() });
			stat.sum += stat_gpu.sum + stat_cpu.Sum();
			stat.sumAbs += stat_gpu.sumAbs + stat_cpu.SumAbs();
		}
		else
		{
			// Find statistics on GPU
			SDataStat stat_gpu = ppr::executor::RunStatisticsOnGPU(opencl, configuration, arena, data);

			// Agregate results results
			stat.n += stat_gpu.n;
			stat.min = stat_gpu.min;
			stat.max = stat_gpu.max;
			stat.sum += stat_gpu.sum;
			stat.sumAbs += stat_gpu.sumAbs;
		}

	}

	void CreateFrequencyHistogram(SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<double>& histogram)
	{
		// Update kernel program
		ppr::gpu::UpdateProgram(opencl, HIST_KERNEL, HIST_KERNEL_NAME);

		SHistogram hist;
		hist.binCount = log2(data_count) + 1;
		hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		ppr::hist::HistogramParallel hist_cpu(hist.binCount, hist.binSize, stat.min, stat.max, data, stat.mean);
		ppr::executor::RunOnCPU<ppr::hist::HistogramParallel>(arena, hist_cpu, opencl.data_count_for_cpu, data_count);

		ppr::executor::RunHistogramOnGPU(opencl, stat, hist, arena, data, hist_cpu.m_bucketFrequency);

		hist_cpu.ComputePropabilityDensityOfHistogram(histogram, data_count);

		double tmp = (stat.variance + hist_cpu.m_var) / stat.n;
		stat.variance = tmp;
	}

	void AnalyzeResults(SResult& res)
	{
		// Find min RSS value
		std::array<double, 4> rss = { res.gauss_rss, res.poisson_rss, res.exp_rss, res.uniform_rss };
		std::sort(rss.begin(), rss.end());

		bool canBePoisson = !res.isNegative && res.isInteger;
		bool canBeExp = !res.isNegative && !res.isInteger;

		if (res.uniform_rss == rss[0])
		{
			res.dist = EDistribution::UNIFORM;
		}
		else if ((res.poisson_rss == rss[0] && canBePoisson) || (res.poisson_rss == rss[1] && canBePoisson))
		{
			res.dist = EDistribution::POISSON;
		}
		else if ((res.exp_rss == rss[0] && canBeExp) || (res.exp_rss == rss[1] && canBeExp))
		{
			res.dist = EDistribution::EXP;
		}
		else if (res.gauss_rss == rss[0])
		{
			res.dist = EDistribution::GAUSS;
		}
		res.status = EExitStatus::SUCCESS;
	}

	void CalculateHistogramRSS(SResult& res, tbb::task_arena& arena, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		ppr::rss::Distribution* gauss = new ppr::rss::NormalDistribution(res.gauss_mean, res.gauss_stdev, res.gauss_variance);
		ppr::rss::Distribution* poisson = new ppr::rss::PoissonDistribution(res.poisson_lambda);
		ppr::rss::Distribution* exp = new ppr::rss::ExponentialDistribution(res.exp_lambda);
		ppr::rss::Distribution* uniform = new ppr::rss::UniformDistribution(res.uniform_a, res.uniform_b);

		ppr::rss::RSSParallel gauss_rss(gauss, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel poisson_rss(poisson, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel exp_rss(exp, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel uniform_rss(uniform, histogramDensity, hist.binSize);

		double t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, gauss_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Gauss RSS:\t" << t << "\tsec." << std::endl;

		t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, poisson_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Poisson RSS:\t" << t << "\tsec." << std::endl;

		t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, exp_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Expon RSS:\t" << t << "\tsec." << std::endl;

		t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, uniform_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Uniform RSS:\t" << t << "\tsec." << std::endl;

		res.gauss_rss = gauss->Get_RSS();
		res.poisson_rss = poisson->Get_RSS();
		res.exp_rss = exp->Get_RSS();
		res.uniform_rss = uniform->Get_RSS();
	}

	SResult run2(SConfig& configuration)
	{
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Init OpenCL]

		SOpenCLConfig opencl = ppr::gpu::Init(configuration, STAT_KERNEL, STAT_KERNEL_NAME);

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		double* data = mapping.GetData();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}

		//  ================ [Allocations]
		SResult res;
		SDataStat stat;
		std::vector<double> tmp(0);
		unsigned int data_count = mapping.GetCount();

		// Get number of data, which we want to process on GPU
		opencl.wg_count = data_count / opencl.wg_size;
		opencl.data_count_for_gpu = data_count - (data_count % opencl.wg_size);

		// The rest of the data we will process on CPU
		opencl.data_count_for_cpu = opencl.data_count_for_gpu + 1;

		//  ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		GetStatistics(configuration, opencl, stat, arena, data_count, data, tmp);
		tbb::tick_count t1 = tbb::tick_count::now();
		std::cout << "Statistics:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		stat.mean = stat.sum / stat.n;

		//  ================ [Create frequency histogram]
		// Update kernel program
		ppr::gpu::UpdateProgram(opencl, HIST_KERNEL, HIST_KERNEL_NAME);

		// Find histogram limits
		SHistogram hist;
		hist.binCount = log2(mapping.GetCount()) + 1;
		hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		std::vector<double> histogramDensity(static_cast<int>(hist.binCount));

		// Create histogram
		t0 = tbb::tick_count::now();
		CreateFrequencyHistogram(configuration, opencl, stat, arena, data_count, data, histogramDensity);
		t1 = tbb::tick_count::now();
		std::cout << "Histogram:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;


		//	================ [Unmap file]
		mapping.UnmapFile();

		//	================ [Fit params]
		res.isNegative = !(std::floor(stat.sum) == std::floor(stat.sumAbs));
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
		CalculateHistogramRSS(res, arena, histogramDensity, hist);

		//	================ [Analyze]
		AnalyzeResults(res);

		std::cout << "Finish." << std::endl;
		return res;
	}
}
