#include <cmath>
#include <vector>
#include "../rss/rss.cpp"

namespace ppr::hist
{
	class Histogram
	{
		private:
			double BinSize;
			double Min;
			std::vector<uint64_t> Buckets;

		public:
			Histogram(std::vector<uint64_t>& buckets, double bin_size, double min)
				: Buckets(buckets), BinSize(bin_size), Min(min)
			{
				/*double bin_count = log2(mapping.get_count()) + 1;
				double bin_size = (stat.Get_Max() - stat.Get_Min()) / bin_count;
				std::vector<uint64_t> buckets(static_cast<int>(bin_count) + 1);*/
			}

            void Push(double x)
            {
				double position = ((x - Min) / BinSize);
				Buckets[static_cast<int>(position)]++;
            }

			const std::vector<uint64_t> Get_buckets() const
			{
				return Buckets;
			}

			double ComputeRssOfHistogram(char dist_val, double mean = 0.0, double stddev = 0.0, double lambda = 0.0, double mu = 0.0, double a = 0.0, double b = 0.0)
			{
				ppr::rss::Distribution* dist;

				switch (dist_val)
				{
					case 'n':
						dist = new ppr::rss::NormalDistribution(mean, stddev);
						break;
					case 'e':
						dist = new ppr::rss::ExponentialDistribution(lambda);
						break;
					case 'p':
						dist = new ppr::rss::PoissonDistribution(mu);
						break;
					case 'u':
						dist = new ppr::rss::UniformDistribution(a, b);
						break;
					default:
						return 0.0;
				}
				for (unsigned int i = 0; i < Buckets.size(); i++)
				{
					double d = (double)Buckets[i];
					dist->Push(d, (i * BinSize));
				}

				double res = dist->Get_RSS();
				delete dist;
				return res;
			}
	};
}