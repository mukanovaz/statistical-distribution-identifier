#include <cmath>
#include <random>
#include <iostream>
#define M_PI			3.14159265358979323846
#define DOUBLE_PI		(2 * M_PI)

namespace ppr::rss
{
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
				double val = bin - Pdf(y_obs);
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
			double Mean, Stddev;

		public:
			NormalDistribution(double mean, double stddev) 
				: Mean(mean), Stddev(stddev)
			{}

			double Pdf(double x) override
			{
				double t1 = 1.0 / sqrt(DOUBLE_PI);
				double t2 = exp(-0.5 * x * x);
				return t1 * t2;
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
				double t1 = -Lambda * x;
				return Lambda * exp(-x);
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
				double t1 = x - A;
				double t2 = B - A;
				return t1 / t2;
			}
	};

	class PoissonDistribution : public Distribution
	{
		private:
			double Mu;

			double Factorial(double n)
			{
				double factorial = 0.0;
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
				double t2 = pow(Mu, x) / Factorial(x);
				return t1 * t2;
			}
	};
}

