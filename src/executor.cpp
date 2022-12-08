#include "include/executor.h"
#include "include/smp_utils.h"
#include <algorithm>

namespace ppr::executor
{
	double sum_vector_tbb(tbb::task_arena& arena, std::vector<double> data)
	{
		double sum = 0;
		arena.execute([&]() {
			sum = tbb::parallel_reduce(tbb::blocked_range<std::vector<double>::iterator>(data.begin(), data.end()), 0.0,
				[](tbb::blocked_range<std::vector<double>::iterator> const& range, double init) {
					return std::accumulate(range.begin(), range.end(), init);
				}, std::plus<double>());
			});

		return sum;
	}

	void calculate_histogram_RSS_with_tbb(SResult& res, tbb::task_arena& arena, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		tbb::tick_count total1 = tbb::tick_count::now();

		ppr::rss::Distribution* gauss = new ppr::rss::NormalDistribution(res.gauss_mean, res.gauss_stdev, res.gauss_variance);
		ppr::rss::Distribution* poisson = new ppr::rss::PoissonDistribution(res.poisson_lambda);
		ppr::rss::Distribution* exp = new ppr::rss::ExponentialDistribution(res.exp_lambda);
		ppr::rss::Distribution* uniform = new ppr::rss::UniformDistribution(res.uniform_a, res.uniform_b);

		ppr::rss::RSSParallel gauss_rss(gauss, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel poisson_rss(poisson, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel exp_rss(exp, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel uniform_rss(uniform, histogramDensity, hist.binSize);

		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, gauss_rss, 0, static_cast<unsigned long long>(hist.binCount));
		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, poisson_rss, 0, static_cast<unsigned long long>(hist.binCount));
		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, exp_rss, 0, static_cast<unsigned long long>(hist.binCount));
		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, uniform_rss, 0, static_cast<unsigned long long>(hist.binCount));

		res.gauss_rss = gauss->Get_RSS();
		res.poisson_rss = poisson->Get_RSS();
		res.exp_rss = exp->Get_RSS();
		res.uniform_rss = uniform->Get_RSS();

		// Free allocations
		delete gauss;
		delete poisson;
		delete exp;
		delete uniform;

		tbb::tick_count total2 = tbb::tick_count::now();
		res.total_rss_time = (total2 - total1).seconds();
	}

	void analyze_results(SResult& res)
	{
		// Find min RSS value
		std::array<double, 4> rss = { res.gauss_rss, res.poisson_rss, res.exp_rss, res.uniform_rss };
		std::sort(rss.begin(), rss.end());

		bool canBePoisson = !res.isNegative && res.isInteger && res.poisson_lambda > 0;
		bool canBeExp = !res.isNegative && !res.isInteger && res.exp_lambda > 0;

		if (canBePoisson)
		{
			if (res.poisson_rss == rss[0] || res.exp_rss == rss[0] || res.poisson_rss == rss[1])
			{
				res.dist = EDistribution::POISSON;
			}
		}
		else if (canBeExp && res.exp_rss == rss[0])
		{
			res.dist = EDistribution::EXP;
		}
		else if(res.gauss_rss == rss[0])
		{
			res.dist = EDistribution::GAUSS;
		}
		else if (res.uniform_rss == rss[0])
		{
			res.dist = EDistribution::UNIFORM;
		}

		res.status = EExitStatus::SUCCESS;
	}

	void compute_propability_density_histogram(SHistogram& hist, std::vector<int>& bucket_frequency, std::vector<double>& bucket_density, unsigned long long count)
	{
		for (size_t i = 0; i < static_cast<size_t>(hist.binCount); i++)
		{
			double next_edge = hist.min + (hist.binSize * (static_cast<double>(i) + 1.0));
			double curr_edge = hist.min + (hist.binSize * static_cast<double>(i));
			double diff = next_edge - curr_edge;
			bucket_density[i] = bucket_frequency[i] / diff / count;
		}
	}
	
}