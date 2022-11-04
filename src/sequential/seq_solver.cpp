#include "seq_solver.h"
#include "../rss/statistics.cpp"
#include "../file_mapping.h"
#include "../file_mapping.cpp"
#include "../rss/rss.cpp"

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

		for (unsigned int i = 1; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			stat.Push(d);
		}

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

		// ================ [Gauss RSS]
		ppr::rss::RSSGauss gauss_rss(gauss_mean, gauss_sd);
		for (unsigned int i = 0; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			gauss_rss.Push(d);
		}
		double g_rss = gauss_rss.Get_RSS();


		// ================ [Exponential RSS]
		ppr::rss::RSSExp exp_rss(exp_lambda);
		for (unsigned int i = 0; i < mapping.get_count(); i++)
		{
			double d = (double)data[i];
			exp_rss.Push(d);
		}
		double e_rss = exp_rss.Get_RSS();

		mapping.unmap_file();

		return SResult::error_res(EExitStatus::STAT);
	}
}