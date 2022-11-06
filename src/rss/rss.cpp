#include<cmath>
#include <random>
#include <iostream>
# define M_PI           3.14159265358979323846

namespace ppr::rss
{



	class NormalDistribution
	{
		private:
			double RSS, Mean, Stddev, BinSize;
			int BinCount;
			std::vector<uint64_t> Buckets;

		public:
			NormalDistribution(const std::vector<uint64_t>& buckets, double binsize, int bincount, double mean, double stddev)
				: RSS(0.0), BinSize(binsize), BinCount(bincount), Buckets(buckets), Mean(mean), Stddev(stddev)
			{}

			double Pdf(double x)
			{
				double t1 = 1.0 / sqrt(2 * M_PI);
				double t2 = exp(-0.5 * x * x);
				return t1 * t2;
			}

			void Push(double y_obs, double bin)
			{
				double val = bin - Pdf(y_obs);
				double tmp = RSS + (val * val);
				RSS = tmp;
			}

			double Get_RSS() const
			{
				return RSS;
			}


	};

	class ExponentialDistribution
	{
		private:
			double RSS, Lambda, Mean, BinSize;
			int BinCount;
			std::vector<uint64_t> Buckets;

		public:
			ExponentialDistribution(const std::vector<uint64_t>& buckets, double binsize, int bincount, double lambda, double mean)
				: RSS(0.0), Lambda(lambda), Mean(mean), BinSize(binsize), BinCount(bincount), Buckets(buckets)
			{}

			double Pdf(double x)
			{
				double t1 = -Lambda * x;
				return Lambda * exp(-x);
			}

			void Push(double y_obs, double bin)
			{
				double val = bin - Pdf(y_obs);
				double tmp = RSS + (val * val);
				RSS = tmp;
			}

			double Get_RSS() const
			{
				return RSS;
			}
	};

}

