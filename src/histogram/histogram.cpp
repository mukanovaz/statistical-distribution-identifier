#include <cmath>
#include <vector>
#include <memory>


#undef min
#undef max

#include <tbb/parallel_for.h>
#include "../rss/rss.cpp"
#include "../data.h"

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
					/*std::cout << bucket_density[i] << std::endl;*/
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

	//template<std::size_t N>
	class HistogramParallel
	{
		private:
			SHistogram m_histogram;
			std::vector<int> m_bucketFrequency;
			std::vector<double> m_bucketDensity;
			const double* m_data;

		public:

			HistogramParallel(int size, double bin_size, double min, double max, const double* data)
				: m_data(data)
			{
				m_bucketDensity.resize(size);
				m_bucketFrequency.resize(size);
				m_histogram.binSize = bin_size;
				m_histogram.min = min;
				m_histogram.max = max;
				m_histogram.size = size;
				m_histogram.scaleFactor = (m_histogram.size) / (m_histogram.max - m_histogram.min);
			}

			HistogramParallel(HistogramParallel& x, tbb::split) 
				:m_data(x.m_data)
			{
				m_bucketDensity.resize(x.m_histogram.size);
				m_bucketFrequency.resize(x.m_histogram.size);
				m_histogram.binSize = x.m_histogram.binSize;
				m_histogram.min = x.m_histogram.min;
				m_histogram.max = x.m_histogram.max;
				m_histogram.size = x.m_histogram.size;
				m_histogram.scaleFactor = (x.m_histogram.size) / (x.m_histogram.max - x.m_histogram.min);
			}
			
			void operator()(const tbb::blocked_range<size_t>& r)
			{
				// Parameters 
				const double* t_data = m_data;
				SHistogram t_histogram = m_histogram;						// to not discard earlier accumulations
				std::vector<int>& t_bucketFrequency = m_bucketFrequency;
				std::vector<double>& t_bucketDensity = m_bucketDensity;

				#pragma loop(ivdep)
				for (size_t i = r.begin(); i != r.end(); ++i)
				{
					double x = (double)t_data[i];
					double position = (x - t_histogram.min) * t_histogram.scaleFactor;
					// TODO: remove if
					if (position == t_bucketFrequency.size())
					{
						position--;
					}
					t_bucketFrequency[static_cast<int>(position)]++;
				}

				m_bucketDensity = t_bucketDensity;
				m_bucketFrequency = t_bucketFrequency;
			}

			void join(const HistogramParallel& y)
			{
				std::transform(m_bucketFrequency.begin(), m_bucketFrequency.end(), y.m_bucketFrequency.begin(), m_bucketFrequency.begin(), std::plus<int>());
			}

			void ComputePropabilityDensityOfHistogram(std::vector<double>& bucket_density, double count)
			{
				#pragma loop(ivdep)
				for (unsigned int i = 0; i < m_histogram.size; i++)
				{
					double next_edge = m_histogram.min + (m_histogram.binSize * (static_cast<double>(i) + 1.0));
					double curr_edge = m_histogram.min + (m_histogram.binSize * static_cast<double>(i));
					double diff = next_edge - curr_edge;
					bucket_density[i] = m_bucketFrequency[i] / diff / count;
				}
			}
	};

}

