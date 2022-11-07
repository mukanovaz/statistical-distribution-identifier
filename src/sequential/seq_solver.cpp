#include "seq_solver.h"
#include "../rss/statistics.cpp"
#include "../file_mapping.h"
#include "../file_mapping.cpp"
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

		ppr::hist::Histogram hist(static_cast<int>(bin_count) + 1, bin_size, stat.Get_Min());
		for (unsigned int i = 0; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			hist.Push(d);
		}
		mapping.unmap_file();

		// Get propability density of histogram
		hist.FindBucketEdges();
		hist.ComputePropabilityDensityOfHistogram(mapping.get_count());

		// Fit params
		// ================ [Gauss maximum likelihood estimators]
		double gauss_mean = stat.Mean();
		double gauss_variance = stat.Variance();
		double gauss_sd = stat.StandardDeviation();

		// ================ [Exponential maximum likelihood estimators]
		double exp_lambda = static_cast<double>(stat.NumDataValues()) / stat.Sum();

		// ================ [Poisson likelihood estimators]
		double poisson_lambda = stat.Mean();

		// ================ [Uniform likelihood estimators]
		double a = stat.Get_Min();
		double b = stat.Get_Max();


		// Calculate PDF for params

		double g_rss = hist.ComputeRssOfHistogram('n', gauss_mean, gauss_sd);
		double e_rss = hist.ComputeRssOfHistogram('e', 0.0, 0.0, exp_lambda);
		double p_rss = hist.ComputeRssOfHistogram('p', 0.0, 0.0, 0.0, poisson_lambda);
		double u_rss = hist.ComputeRssOfHistogram('u', 0.0, 0.0, 0.0, 0.0, a, b);


		return SResult::error_res(EExitStatus::STAT);
	}
}