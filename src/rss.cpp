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
			double m_rss;
		
		public:
			Distribution() : m_rss(0.0)
			{}

			virtual double Pdf(double) { return 0.0; }

			void Push(double y_obs, double bin)
			{
				double val = y_obs - Pdf(bin);
				double tmp = m_rss + (val * val);
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
		private:
			double Mean, Stddev, Variance;

		public:
			NormalDistribution(double mean, double stddev, double variance)
				: Mean(mean), Stddev(stddev), Variance(variance)
			{}

			double Pdf(double x) override
			{
				return 1.0 / sqrt(DOUBLE_PI) * exp( -0.5 * x * x);
				//return 1.0 / sqrt(DOUBLE_PI * Variance) * exp( -0.5 * pow((x - Mean)/ Stddev, 2));
			}
	};

	class ExponentialDistribution : public Distribution
	{
		private:
			double Lambda;

		public:
			ExponentialDistribution(double lambda)
				: Lambda(lambda)
			{}

			virtual double Pdf(double x) override
			{
				// TODO: x >= 0
				if (x < 0) return 0;
				double betta = 1 / Lambda;
				double t1 = 1 / betta;
				double t2 = exp(-(x / betta));
				return t1 * t2;
			}
	};

	class UniformDistribution : public Distribution
	{
		private:
			double A, B;

		public:
			UniformDistribution(double a, double b)
				: A(a), B(b)
			{}

			double Pdf(double) override
			{
				// TODO: a <= x <= b
				return 1.0 / (B - A);
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
				// TODO: x >= 0; Mu >= 0; If Mu == 0 -> PDF = 1.0
				double t1 = exp(-Mu);
				double t2 = pow(Mu, x);
				return (t1 * t2) / Factorial(x);
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

