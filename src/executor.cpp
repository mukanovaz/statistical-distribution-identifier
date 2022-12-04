#include "executor.h"
#include "smp/smp_utils.h"
#include <algorithm>

namespace ppr::executor
{
	// https://www.cs.cmu.edu/afs/cs/academic/class/15499-s09/www/handouts/TBB-HPCC07.pdf
	class MinParallel {
		const std::vector<double> my_a;
	public:
		double value_of_min;

		MinParallel(const std::vector<double> a) :
			my_a(a),
			value_of_min(FLT_MAX)
		{}

		MinParallel(MinParallel& x, tbb::split) :
			my_a(x.my_a),
			value_of_min(FLT_MAX)
		{}

		void operator()(const tbb::blocked_range<size_t>& r) {
			const std::vector<double> a = my_a;
			for (size_t i = r.begin(); i != r.end(); ++i) {
				double value = a[i];
				value_of_min = std::min({ value_of_min , value });
			}
		}

		void join(const MinParallel& y) {
			value_of_min = std::min({ value_of_min , y.value_of_min });
		}
	};

	class MaxParallel {
		const std::vector<double> my_a;
	public:
		double value_of_max;

		MaxParallel(const std::vector<double> a) :
			my_a(a),
			value_of_max(FLT_MIN)
		{}

		MaxParallel(MaxParallel& x, tbb::split) :
			my_a(x.my_a),
			value_of_max(FLT_MIN)
		{}

		void operator()(const tbb::blocked_range<size_t>& r) {
			const std::vector<double> a = my_a;
			for (size_t i = r.begin(); i != r.end(); ++i) {
				double value = a[i];
				value_of_max = std::max({ value_of_max , value });
			}
		}

		void join(const MaxParallel& y) {
			value_of_max = std::max({ value_of_max , y.value_of_max });
		}
	};

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

		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, gauss_rss, 0, static_cast<int>(hist.binCount));
		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, poisson_rss, 0, static_cast<int>(hist.binCount));
		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, exp_rss, 0, static_cast<int>(hist.binCount));
		ppr::executor::run_with_tbb<ppr::rss::RSSParallel>(arena, uniform_rss, 0, static_cast<int>(hist.binCount));

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

	void compute_propability_density_histogram(SHistogram& hist, std::vector<int>& bucket_frequency, std::vector<double>& bucket_density, double count)
	{
		for (unsigned int i = 0; i < hist.binCount; i++)
		{
			double next_edge = hist.min + (hist.binSize * (static_cast<double>(i) + 1.0));
			double curr_edge = hist.min + (hist.binSize * static_cast<double>(i));
			double diff = next_edge - curr_edge;
			bucket_density[i] = bucket_frequency[i] / diff / count;
		}
	}
	
}