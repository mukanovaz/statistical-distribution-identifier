#include "include/seq_solver.h"

namespace ppr::seq
{
	
	SResult run(SConfig& configuration)
	{
		tbb::tick_count total1;
		tbb::tick_count total2;
		total1 = tbb::tick_count::now();
		SResult res;
		File_mapping mapping(configuration.input_fn);

		const double* data = mapping.get_data();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}

		RunningStat stat(data[0]);

		// ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		for (long i = 1; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			stat.Push(d);
		}
		tbb::tick_count t1 = tbb::tick_count::now();
		res.total_stat_time = (t1 - t0).seconds();

		//	================ [Fit params]
		res.isNegative = stat.Get_Min() < 0;
		res.isInteger = std::floor(stat.Sum()) == stat.Sum();

		// Gauss maximum likelihood estimators
		res.gauss_mean = stat.Mean();
		res.gauss_variance = stat.Variance();
		res.gauss_stdev = stat.StandardDeviation();

		// Exponential maximum likelihood estimators
		res.exp_lambda = static_cast<double>(stat.NumDataValues()) / stat.Sum();;

		// Poisson likelihood estimators
		res.poisson_lambda = stat.Sum() / stat.NumDataValues();

		// Uniform likelihood estimators
		res.uniform_a = stat.Get_Min();
		res.uniform_b = stat.Get_Max();

		// ================ [Create histogram]
		t0 = tbb::tick_count::now();

		double bin_count = 0.0;
		double bin_size = 0.0;

		if (!res.isNegative && res.isInteger && res.poisson_lambda > 0)
		{
			bin_count = stat.Get_Max() - stat.Get_Min();
			bin_size = 1.0;
		}
		else
		{
			bin_count = log2(stat.NumDataValues()) + 2;
			bin_size = (stat.Get_Max() - stat.Get_Min()) / (bin_count - 1);
		}

		ppr::hist::Histogram hist(static_cast<int>(bin_count), bin_size, stat.Get_Min(), stat.Get_Max());

		std::vector<int> histogramFrequency(static_cast<int>(bin_count) + 1);
		std::vector<double> histogramDensity(static_cast<int>(bin_count) + 1);

		for (unsigned int i = 0; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			hist.push(histogramFrequency, d);
		}

		t1 = tbb::tick_count::now();
		res.total_hist_time = (t1 - t0).seconds();

		mapping.unmap_file();

		// ================ [Get propability density of histogram]
		hist.compute_propability_density_histogram(histogramDensity, histogramFrequency, mapping.get_count());

		// ================ [Calculate RSS]
		t0 = tbb::tick_count::now();
		res.gauss_rss = hist.compute_rss_histogram(histogramDensity, 'n', res);
		res.exp_rss = hist.compute_rss_histogram(histogramDensity, 'e', res);
		res.poisson_rss = hist.compute_rss_histogram(histogramDensity, 'p', res);
		res.uniform_rss = hist.compute_rss_histogram(histogramDensity, 'u', res);
		t1 = tbb::tick_count::now();
		res.total_rss_time = (t1 - t0).seconds();

		//	================ [Analyze]
		ppr::executor::analyze_results(res);

		total2 = tbb::tick_count::now();
		res.total_time = (total2 - total1).seconds();

		std::cout << "\t\t\t[Statistics]" << std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;
		std::cout << "> n:\t\t\t\t" << stat.NumDataValues() << std::endl;
		std::cout << "> sum:\t\t\t\t" << stat.Sum() << std::endl;
		std::cout << "> mean:\t\t\t\t" << stat.Mean() << std::endl;
		std::cout << "> variance:\t\t\t" << stat.Variance() << std::endl;
		std::cout << "> min:\t\t\t\t" << stat.Get_Min() << std::endl;
		std::cout << "> max:\t\t\t\t" << stat.Get_Max() << std::endl;

		return res;
	}
}