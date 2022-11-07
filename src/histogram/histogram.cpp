#include <cmath>
#include <vector>
#include <memory>
#include "../rss/rss.cpp"

namespace ppr::hist
{
	class Histogram
	{
		private:
			double BinSize;
			double Min, Size;
			std::unique_ptr<double[]> BucketsDensity;
			std::unique_ptr<double[]> BucketFrequency;
			std::unique_ptr<double[]> BucketEdges;

		public:
			Histogram(int size, double bin_size, double min)
				: Size(size), BinSize(bin_size), Min(min)
			{
				BucketFrequency = std::make_unique<double[]>(size);
				BucketsDensity = std::make_unique<double[]>(size);
				BucketEdges = std::make_unique<double[]>(size);
				//Buckets.reserve()
				/*double bin_count = log2(mapping.get_count()) + 1;
				double bin_size = (stat.Get_Max() - stat.Get_Min()) / bin_count;
				std::vector<uint64_t> buckets(static_cast<int>(bin_count) + 1);*/
			}

            void Push(double x)
            {
				double position = ((x - Min) / BinSize);
				BucketFrequency[static_cast<int>(position)]++;
            }

			void FindBucketEdges()
			{
				for (unsigned int i = 0; i < Size; i++)
				{
					BucketEdges[i] = BinSize * i;
					//std::cout << BucketEdges[i] << std::endl;;
				}
			}

			void ComputePropabilityDensityOfHistogram(double count)
			{
				for (unsigned int i = 0; i < Size - 1; i++)
				{
					double diff = BucketEdges[i + 1] - BucketEdges[i];
					BucketsDensity[i] = BucketFrequency[i] / diff/ count;
					std::cout << BucketsDensity[i] << std::endl;;
				}
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
				for (unsigned int i = 0; i < Size; i++)
				{
					double d = (double)BucketsDensity[i];
					dist->Push(d, (i * BinSize));
				}
				
				double res = dist->Get_RSS();
				delete dist;
				return res;
			}
	};
}