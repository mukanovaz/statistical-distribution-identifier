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

	void analyze_results(SResult& res)
	{
		// Find min RSS value
		std::array<double, 4> rss = { res.gauss_rss, res.poisson_rss, res.exp_rss, res.uniform_rss };
		std::sort(rss.begin(), rss.end());

		if (res.poisson_rss == rss[0])
		{
			res.dist = EDistribution::POISSON;
		}
		else if (res.exp_rss == rss[0])
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
		int binCount = hist.binCount;
		double n = static_cast<double>(count);

		for (int i = 0; i < binCount; i++)
		{
			double next_edge = hist.min + (hist.binSize * (static_cast<double>(i) + 1.0));
			double curr_edge = hist.min + (hist.binSize * static_cast<double>(i));

			double diff = next_edge - curr_edge;
			bucket_density[i] = (double)bucket_frequency[i] / diff / n;
		}
	}
	
}