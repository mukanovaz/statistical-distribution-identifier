#include "gpu_solver.h"
#include "gpu_utils.h"
#include "../file_mapping.h"
#include "../executor.h"


namespace ppr::gpu
{
	SResult run(SConfig& configuration)
	{
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		tbb::tick_count t0 = tbb::tick_count::now();
		//  ================ [Init OpenCL]
		
		SOpenCLConfig opencl = ppr::gpu::Init(configuration, STAT_KERNEL, STAT_KERNEL_NAME);

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		//mapping.ReadInChunks(L"D:/Study/ZCU/5.semestr/PPR/kiv-ppr/referencni_rozdeleni/gauss");

		double* data = mapping.GetData();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}
		
		//  ================ [Get statistics]

		// Get number of data, which we want to process on GPU
		opencl.wg_count = mapping.GetCount() / opencl.wg_size;
		opencl.data_count_for_gpu = mapping.GetCount() - (mapping.GetCount() % opencl.wg_size);

		// The rest of the data we will process on CPU
		unsigned long data_count_for_cpu = opencl.data_count_for_gpu + 1;

		// Find rest of a statistics on CPU
		RunningStatParallel stat_cpu(data, data_count_for_cpu);
		ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat_cpu, data_count_for_cpu, mapping.GetCount());

		// Find statistics on GPU
		SDataStat stat_gpu = ppr::executor::RunStatisticsOnGPU(opencl, configuration, arena, data);

		stat_gpu.n += stat_cpu.NumDataValues();
		stat_gpu.min = std::min({ stat_gpu.min, stat_cpu.Get_Min()});
		stat_gpu.max = std::max({ stat_gpu.max, stat_cpu.Get_Max()});
		stat_gpu.sum += stat_cpu.Sum();
		stat_gpu.sumAbs += stat_cpu.SumAbs();
		stat_gpu.mean = stat_gpu.sum / stat_gpu.n;

		tbb::tick_count t1 = tbb::tick_count::now();
		std::cout << (t1 - t0).seconds() << "sec." << std::endl;

		//  ================ [Create frequency histogram]
		// Update kernel program
		ppr::gpu::UpdateProgram(opencl, HIST_KERNEL, HIST_KERNEL_NAME);

		SHistogram hist;
		hist.binCount = log2(mapping.GetCount()) + 1;
		hist.binSize = (stat_gpu.max - stat_gpu.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat_gpu.max - stat_gpu.min);

		std::vector<double> histogramDensity(static_cast<int>(hist.binCount));

		ppr::hist::HistogramParallel hist_cpu(hist.binCount, hist.binSize, stat_gpu.min, stat_gpu.max, data, stat_gpu.mean);
		ppr::executor::RunOnCPU<ppr::hist::HistogramParallel>(arena, hist_cpu, data_count_for_cpu, mapping.GetCount());
		
		ppr::executor::RunHistogramOnGPU(opencl, stat_gpu, hist, arena, data, hist_cpu.m_bucketFrequency);

		//	================ [Unmap file]
		mapping.UnmapFile();

		//  ================ [Get propability density of histogram]
		hist_cpu.ComputePropabilityDensityOfHistogram(histogramDensity, mapping.GetCount());

		//  ================ [Fit params]
	

		double tmp = (stat_gpu.variance + hist_cpu.m_var) / stat_gpu.n;
		stat_gpu.variance = tmp;

		////	================ [Calculate RSS]
		//ppr::rss::Distribution* gauss = new ppr::rss::NormalDistribution(gauss_mean, gauss_sd, gauss_variance);
		//ppr::rss::Distribution* poisson = new ppr::rss::PoissonDistribution(poisson_lambda);
		//ppr::rss::Distribution* exp = new ppr::rss::ExponentialDistribution(exp_lambda);
		//ppr::rss::Distribution* uniform = new ppr::rss::UniformDistribution(a, b);

		//ppr::rss::RSSParallel gauss_rss(gauss, histogramDensity, bin_size);
		//ppr::rss::RSSParallel poisson_rss(poisson, histogramDensity, bin_size);
		//ppr::rss::RSSParallel exp_rss(exp, histogramDensity, bin_size);
		//ppr::rss::RSSParallel uniform_rss(uniform, histogramDensity, bin_size);

		//t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, gauss_rss, 0, static_cast<int>(bin_count));
		//std::cout << "Gauss RSS: " << t1 << "sec." << std::endl;

		//t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, poisson_rss, 0, static_cast<int>(bin_count));
		//std::cout << "Poisson RSS: " << t1 << "sec." << std::endl;

		//t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, exp_rss, 0, static_cast<int>(bin_count));
		//std::cout << "Exponential RSS: " << t1 << "sec." << std::endl;

		//t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, uniform_rss, 0, static_cast<int>(bin_count));
		//std::cout << "Uniform RSS: " << t1 << "sec." << std::endl;

		//double g_rss = gauss->Get_RSS();
		//double p_rss = poisson->Get_RSS();
		//double e_rss = exp->Get_RSS();
		//double u_rss = uniform->Get_RSS();

		//double t_min = std::min({ g_rss, e_rss, p_rss, u_rss });

		//if (g_rss == t_min)
		//{
		//	std::cout << "This is Gauss" << std::endl;
		//}
		//else if (e_rss == t_min)
		//{
		//	std::cout << "This is Exponential" << std::endl;
		//}
		//else if (p_rss == t_min)
		//{
		//	std::cout << "This is Poisson" << std::endl;
		//}
		//else if (u_rss == t_min)
		//{
		//	std::cout << "This is Uniform" << std::endl;
		//}
		return SResult::error_res(EExitStatus::STAT);
	}
}
