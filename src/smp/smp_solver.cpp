#include "smp_solver.h"
#include "../file_mapping.h"
#include "../executor.h"

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

		double t1 = ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat, 1, mapping.GetCount());
		std::cout << "Get statistics: " << t1 << "sec." << std::endl;

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

		//  ================ [Create frequency histogram]
		const double bin_count = log2(mapping.GetCount()) + 1;
		double bin_size = (stat.Get_Max() - stat.Get_Min()) / (bin_count - 1); // TODO
		std::vector<double> histogramDensity(static_cast<int>(bin_count));

		ppr::hist::HistogramParallel hist(static_cast<int>(bin_count), bin_size, stat.Get_Min(), stat.Get_Max(), data);

		t1 = ppr::executor::RunOnCPU<ppr::hist::HistogramParallel>(arena, hist, 0, mapping.GetCount());
		std::cout << "Histogram: " << t1 << "sec." << std::endl;

		//================ [Unmap file]
		mapping.UnmapFile();

		//  ================ [Get propability density of histogram]
		hist.ComputePropabilityDensityOfHistogram(histogramDensity, mapping.GetCount());

		// ================ [Calculate RSS]
		ppr::rss::Distribution* gauss = new ppr::rss::NormalDistribution(gauss_mean, gauss_sd, gauss_variance);
		ppr::rss::Distribution* poisson = new ppr::rss::PoissonDistribution(poisson_lambda);
		ppr::rss::Distribution* exp = new ppr::rss::ExponentialDistribution(exp_lambda);
		ppr::rss::Distribution* uniform = new ppr::rss::UniformDistribution(a, b);

		ppr::rss::RSSParallel gauss_rss(gauss, histogramDensity, bin_size);
		ppr::rss::RSSParallel poisson_rss(poisson, histogramDensity, bin_size);
		ppr::rss::RSSParallel exp_rss(exp, histogramDensity, bin_size);
		ppr::rss::RSSParallel uniform_rss(uniform, histogramDensity, bin_size);

		t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, gauss_rss, 0, static_cast<int>(bin_count));
		std::cout << "Gauss RSS: " << t1 << "sec." << std::endl;

		t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, poisson_rss, 0, static_cast<int>(bin_count));
		std::cout << "Poisson RSS: " << t1 << "sec." << std::endl;

		t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, exp_rss, 0, static_cast<int>(bin_count));
		std::cout << "Exponential RSS: " << t1 << "sec." << std::endl;

		t1 = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, uniform_rss, 0, static_cast<int>(bin_count));
		std::cout << "Uniform RSS: " << t1 << "sec." << std::endl;

		double g_rss = gauss->Get_RSS();
		double p_rss = poisson->Get_RSS();
		double e_rss = exp->Get_RSS();
		double u_rss = uniform->Get_RSS();

		double t_min = std::min({ g_rss, e_rss, p_rss, u_rss });

		if (g_rss == t_min)
		{
			std::cout << "This is Gauss" << std::endl;
		}
		else if (e_rss == t_min)
		{
			std::cout << "This is Exponential" << std::endl;
		}
		else if (p_rss == t_min)
		{
			std::cout << "This is Poisson" << std::endl;
		}
		else if (u_rss == t_min)
		{
			std::cout << "This is Uniform" << std::endl;
		}
		return SResult::error_res(EExitStatus::STAT);
	}

}