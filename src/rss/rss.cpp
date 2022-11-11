#include <cmath>
#include <iostream>

namespace ppr::rss
{
	constexpr double M_PI = 3.14159265358979323846;
	constexpr double DOUBLE_PI = (2 * M_PI);

	class Distribution
	{
		protected:
			double RSS;
		
		public:
			Distribution() : RSS(0.0)
			{}

			virtual double Pdf(double x) { return 0.0; }

			void Push(double y_obs, double bin)
			{
				double val = y_obs - Pdf(bin);
				double tmp = RSS + (val * val);
				RSS = tmp;
			}

			double Get_RSS() const
			{
				return RSS;
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

			double Pdf(double x) override
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

			double Pdf(double x) override
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
}

