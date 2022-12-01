#include "seq_solver.h"

namespace ppr::seq
{
	
	SResult run(SConfig& configuration)
	{
		tbb::tick_count total0 = tbb::tick_count::now();
		SResult res;
		FileMapping mapping(configuration.input_fn);

		const double* data = mapping.GetData();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}

		RunningStat stat(data[0]);

		// ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		for (unsigned int i = 1; i < mapping.GetCount(); i++)
		{
			double d = (double)data[i];
			stat.Push(d);
		}
		tbb::tick_count t1 = tbb::tick_count::now();
		std::cout << "Statistics:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		// ================ [Create histogram]
		t0 = tbb::tick_count::now();

		const double bin_count = log2(mapping.GetCount()) + 1;
		double bin_size = (stat.Get_Max() - stat.Get_Min()) / (bin_count - 1); // TODO

		ppr::hist::Histogram hist(static_cast<int>(bin_count), bin_size, stat.Get_Min(), stat.Get_Max());

		std::vector<int> histogramFrequency(static_cast<int>(bin_count) + 1);
		std::vector<double> histogramDensity(static_cast<int>(bin_count) + 1);

		for (unsigned int i = 0; i < mapping.GetCount(); i++)
		{
			double d = (double)data[i];
			hist.Push(histogramFrequency, d);
		}

		t1 = tbb::tick_count::now();
		std::cout << "Histogram:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;
		mapping.UnmapFile();

		// ================ [Get propability density of histogram]
		hist.ComputePropabilityDensityOfHistogram(histogramDensity, histogramFrequency, mapping.GetCount());

		//	================ [Fit params]
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

		// ================ [Calculate RSS]
		t0 = tbb::tick_count::now();
		res.gauss_rss = hist.ComputeRssOfHistogram(histogramDensity, 'n', res);
		res.exp_rss = hist.ComputeRssOfHistogram(histogramDensity, 'e', res);
		res.poisson_rss = hist.ComputeRssOfHistogram(histogramDensity, 'p', res);
		res.uniform_rss = hist.ComputeRssOfHistogram(histogramDensity, 'u', res);
		t1 = tbb::tick_count::now();
		std::cout << "Total RSS:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		//	================ [Analyze]
		ppr::executor::AnalyzeResults(res);

		tbb::tick_count total1 = tbb::tick_count::now();
		std::cout << "Total:\t" << (total1 - total0).seconds() << "\tsec." << std::endl;

		return res;
	}
}