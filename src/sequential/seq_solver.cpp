#include "seq_solver.h"
#include "../rss/statistics.cpp"
#include "../file_mapping.h"
#include "../file_mapping.cpp"
#include "../rss/rss.cpp"
#include "../histogram/histogram.cpp"

namespace ppr::seq
{
	SResult run(SConfig& configuration)
	{
		SResult result;
		FileMapping mapping(configuration.input_fn);

		const double* data = mapping.get_data();

		if (!data)
		{
			return SResult::error_res(EExitStatus::STAT);
		}

		RunningStat stat(data[0]);

		// Get statistics
		for (unsigned int i = 1; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			stat.Push(d);
		}

		// Create histogram
		double bin_count = log2(mapping.get_count()) + 1;
		double bin_size = (stat.Get_Max() - stat.Get_Min()) / bin_count;
		std::vector<uint64_t> buckets(static_cast<int>(bin_count) + 1);

		ppr::hist::Histogram hist(buckets, bin_size, stat.Get_Min());
		for (unsigned int i = 0; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			hist.Push(d);
		}
		buckets = hist.Get_buckets();
		mapping.unmap_file();

		// Fit params
		// ================ [Gauss maximum likelihood estimators]
		double gauss_mean = stat.Mean();
		double gauss_variance = stat.Variance();
		double gauss_sd = stat.StandardDeviation();

		// ================ [Exponential maximum likelihood estimators]
		double exp_lambda = static_cast<double>(stat.NumDataValues()) / stat.Sum();
		double exp_mean = 1.0 / exp_lambda;

		// ================ [Poisson likelihood estimators]
		double poisson_lambda = stat.Mean();

		// ================ [Uniform likelihood estimators]
		double a = stat.Get_Min();
		double b = stat.Get_Max();


		// Calculate PDF for params

		// ================ [Gauss RSS]
		ppr::rss::NormalDistribution gauss_rss(buckets, bin_size, buckets.size(), gauss_mean, gauss_sd);
		for (unsigned int i = 0; i < buckets.size(); i++)
		{
			double d = (double)buckets[i];
			gauss_rss.Push(d, (i * bin_size));
		}
		double g_rss = gauss_rss.Get_RSS();


		// ================ [Exponential RSS]
		ppr::rss::ExponentialDistribution exp_rss(buckets, bin_size, buckets.size(), exp_lambda, exp_mean);
		for (unsigned int i = 0; i < buckets.size(); i++)
		{
			double d = (double)buckets[i];
			exp_rss.Push(d, (i * bin_size));
		}
		double e_rss = exp_rss.Get_RSS();


		return SResult::error_res(EExitStatus::STAT);
	}
}