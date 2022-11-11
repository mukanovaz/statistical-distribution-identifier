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
			double Max, Min, Size, ScaleFactor;

		public:
			Histogram(int size, double bin_size, double min, double max)
				: Size(size), BinSize(bin_size), Min(min), Max(max), ScaleFactor(0.0)
			{
				ScaleFactor = (Size) / (Max - Min);
			}

			void Push(std::vector<int>& arr, double x)
			{
				double position = (x - Min) * ScaleFactor;
				if (position == arr.size())
				{
					position--;
				}
				arr[static_cast<int>(position)]++;
			}

			void ComputePropabilityDensityOfHistogram(std::vector<double>& bucket_density, std::vector<int>& bucket_frequency, double count)
			{
				for (unsigned int i = 0; i < Size; i++)
				{
					std::cout << bucket_frequency[i] << std::endl;
				}

				for (unsigned int i = 0; i < Size; i++)
				{
					double next_edge = Min + (BinSize * (static_cast<double>(i) + 1.0));
					double curr_edge = Min + (BinSize * static_cast<double>(i));
					double diff = next_edge - curr_edge;
					bucket_density[i] = bucket_frequency[i] / diff / count;
					std::cout << bucket_density[i] << std::endl;
				}
			}

			double ComputeRssOfHistogram(std::vector<double>& bucket_density, char dist_val, double variance = 0.0, double mean = 0.0, double stddev = 0.0, double lambda = 0.0, double mu = 0.0, double a = 0.0, double b = 0.0)
			{
				ppr::rss::Distribution* dist;

				switch (dist_val)
				{
					case 'n':
						dist = new ppr::rss::NormalDistribution(mean, stddev, variance);
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
					double d = (double)bucket_density[i];
					dist->Push(d, (i * BinSize));
				}
				
				double res = dist->Get_RSS();
				delete dist;
				return res;
			}
	};
}