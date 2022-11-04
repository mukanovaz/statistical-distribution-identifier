#include<cmath>
#include <random>
#include <iostream>

namespace ppr::rss
{
	class RSSGauss
	{
		private:
			std::default_random_engine Generator;
			double RSS, Mean, Stddev;
			std::normal_distribution<double> Gauss;

		public:
			RSSGauss(double mean, double stddev)
				: RSS(0.0), Mean(mean), Stddev(stddev)
			{
				Gauss = std::normal_distribution<double>{ Mean, Stddev };
			}

			void Push(double x)
			{
				double rand = Gauss(Generator);
				//std::cout << rand << std::endl;
				double val = x - rand;
				double tmp = RSS + (val * val);
				RSS = tmp;
			}

			double Get_RSS() const
			{
				return RSS;
			}


	};

	class RSSExp
	{
		private:
			std::default_random_engine Generator;
			double RSS, Lambda;
			std::exponential_distribution<double> Exp;

		public:
			RSSExp(double lambda)
				: RSS(0.0), Lambda(lambda)
			{
				Exp = std::exponential_distribution<double>{ lambda };
			}

			void Push(double x)
			{
				double rand = Exp(Generator);
				//std::cout << rand << std::endl;
				double val = x - rand;
				double tmp = RSS + (val * val);
				RSS = tmp;
			}

			double Get_RSS() const
			{
				return RSS;
			}
	};

}

