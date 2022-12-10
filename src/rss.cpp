#include <cmath>
#include <iostream>

#undef min
#undef max

#include <tbb/parallel_for.h>

namespace ppr::rss
{
	constexpr double M_PI = 3.14159265358979323846;
	constexpr double DOUBLE_PI = (2 * M_PI);

	class Distribution
	{
		protected:
			double m_rss, m_mean, m_stddev;
		
		public:

			Distribution() : m_rss(0.0), m_mean(0.0), m_stddev(0.0)
			{}

			Distribution(double mean, double stdev) : m_rss(0.0), m_mean(mean), m_stddev(stdev)
			{}

			virtual double Pdf(double) { return 0.0; }

			void Push(double density_x, double bin)
			{
				double pdf = 0.0;
				double tmp = 0.0;

				if (m_stddev == 0.0 && m_mean == 0.0)
				{
					pdf = Pdf(bin);
					double val = pow(density_x - pdf, 2);
					tmp = m_rss + val;
				}
				else
				{
					double y = (bin - m_mean) / m_stddev;
					pdf = Pdf(y) / m_stddev;

					double val = pow(density_x - pdf, 2);
					tmp = m_rss + val;
				}
				m_rss = tmp;
			}

			double Get_RSS() const
			{
				return m_rss;
			}
			
			void Add_RSS(double rss)
			{
				m_rss += rss;
			}
	};


	class NormalDistribution : public Distribution
	{
		public:
			using Distribution::Distribution;

			double Pdf(double x) override
			{
				return exp(-pow(x, 2) / 2.0) / sqrt(DOUBLE_PI);
			}
	};

	class ExponentialDistribution : public Distribution
	{
		public:
			using Distribution::Distribution;

			virtual double Pdf(double x) override
			{
				if (x >= 0.0)
					return exp(-x);
				else
					return 0.0;
			}
	};

	class UniformDistribution : public Distribution
	{
		private:
			double A, B;

		public:
			using Distribution::Distribution;

			double Pdf(double) override
			{
				return 1.0;
			}
	};

	class PoissonDistribution : public Distribution
	{
		private:
			double Mu;

			double Factorial(double n)
			{
				double factorial = 1.0;
				for (int i = 1; i <= n; ++i) {
					factorial *= i;
				}
				return factorial;
			}

		public:
			PoissonDistribution(double mu)
				: Mu(mu)
			{}

			double Pdf(double x) override
			{
				if (x >= 0.0)
					return exp(-Mu) * pow(Mu, x) / Factorial(x);
				else
					return 0.0;
			}
	};

	class RSSParallel {

	private:
		ppr::rss::Distribution* m_dist;
		std::vector<double> m_bucketDensity;
		double m_bin_size;

	public:

		RSSParallel(ppr::rss::Distribution* dist, std::vector<double>& bucketDensity, double bin_size)
			: m_dist(dist), m_bucketDensity(bucketDensity), m_bin_size(bin_size)
		{}

		RSSParallel(RSSParallel& x, tbb::split)
		{
			m_dist = x.m_dist;
			m_bucketDensity = x.m_bucketDensity;
			m_bin_size = x.m_bin_size;
		}

		void operator()(const tbb::blocked_range<size_t>& r)
		{
			ppr::rss::Distribution* t_dist = m_dist;    // to not discard earlier accumulations

			#pragma loop(ivdep)
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				double d = (double)m_bucketDensity[i];
				m_dist->Push(d, (static_cast<double>(i) * m_bin_size));
			}

			m_dist = t_dist;
		}

		void join(const RSSParallel& y)
		{
			m_dist->Add_RSS(y.m_dist->Get_RSS());
		}
	};
}

