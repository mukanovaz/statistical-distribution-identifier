#include "include/histogram.h"

namespace ppr::hist
{
	Histogram::Histogram(int size, double bin_size, double min, double max)
		: Size(size), BinSize(bin_size), Min(min), Max(max), ScaleFactor(0.0)
	{
		ScaleFactor = (Size) / (Max - Min);
	}

	void Histogram::push(std::vector<int>& arr, double x)
	{
		double position = (x - Min) * ScaleFactor;
		if (position == arr.size())
		{
			position--;
		}
		// Update histogram
		arr[static_cast<int>(position)]++;
	}

	void Histogram::compute_propability_density_histogram(std::vector<double>& bucket_density, std::vector<int>& bucket_frequency, double count)
	{
		for (unsigned int i = 0; i < Size; i++)
		{
			double next_edge = Min + (BinSize * (static_cast<double>(i) + 1.0));
			double curr_edge = Min + (BinSize * static_cast<double>(i));
			double diff = next_edge - curr_edge;
			bucket_density[i] = bucket_frequency[i] / diff / count;
		}
	}

	double Histogram::compute_rss_histogram(std::vector<double>& bucket_density, char dist_val, SResult& res)
	{
		ppr::rss::Distribution* dist;

		// Create distribution instance
		switch (dist_val)
		{
		case 'n':
			dist = new ppr::rss::NormalDistribution(res.gauss_mean, res.gauss_stdev);
			break;
		case 'e':
			dist = new ppr::rss::ExponentialDistribution(res.uniform_a, res.gauss_mean - res.uniform_a);
			break;
		case 'p':
			dist = new ppr::rss::PoissonDistribution(res.poisson_lambda);
			break;
		case 'u':
			dist = new ppr::rss::UniformDistribution(res.uniform_a, res.uniform_b - res.uniform_a);
			break;
		default:
			return 0.0;
		}

		// Compute RSS
		for (unsigned int i = 0; i < Size; i++)
		{
			double d = (double)bucket_density[i];
			dist->Push(d, (res.uniform_a + (i * BinSize)));
		}

		double result = dist->Get_RSS();
		delete dist;
		return result;
	}

	Histogram_parallel::Histogram_parallel(int size, double bin_size, double min, double max, const double* data, double mean)
		: m_data(data), m_mean(mean), m_var(0.0)
	{
		m_bucketFrequency.resize(static_cast<size_t>(size) + 1);
		m_histogram.binSize = bin_size;
		m_histogram.min = min;
		m_histogram.max = max;
		m_histogram.binCount = size + 1;
		m_histogram.scaleFactor = (m_histogram.binCount) / (m_histogram.max - m_histogram.min);
	}

	Histogram_parallel::Histogram_parallel(Histogram_parallel& x, tbb::split)
		:m_data(x.m_data), m_mean(x.m_mean), m_var(0.0)
	{
		m_bucketFrequency.resize(static_cast<size_t>(x.m_histogram.binCount));
		m_histogram.binSize = x.m_histogram.binSize;
		m_histogram.min = x.m_histogram.min;
		m_histogram.max = x.m_histogram.max;
		m_histogram.binCount = x.m_histogram.binCount;
		m_histogram.scaleFactor = (x.m_histogram.binCount) / (x.m_histogram.max - x.m_histogram.min);
	}

	void Histogram_parallel::operator()(const tbb::blocked_range<size_t>& r)
	{
		// Parameters 
		double t_var = m_var;
		const double* t_data = m_data;
		SHistogram t_histogram = m_histogram;						// to not discard earlier accumulations
		std::vector<int>& t_bucketFrequency = m_bucketFrequency;
		size_t begin = r.begin();
		size_t end = r.end();

		for (size_t i = begin; i != end; i++)
		{
			// Get new position
			double x = (double)t_data[i];
			int position = static_cast<int>((x - t_histogram.min) * t_histogram.scaleFactor);
			position = position == t_bucketFrequency.size() ? position - 1 : position;

			// Update histogram
			t_bucketFrequency[position]++;

			// Compute part of variance
			double tmp = x - m_mean;
			t_var += tmp * tmp;
		}
		m_var = t_var;
		m_bucketFrequency = t_bucketFrequency;
	}

	void Histogram_parallel::join(const Histogram_parallel& y)
	{
		// Accumulate values and histogram
		m_var += y.m_var;
		std::transform(m_bucketFrequency.begin(), m_bucketFrequency.end(), y.m_bucketFrequency.begin(), m_bucketFrequency.begin(), std::plus<int>());
	}

	void Histogram_parallel::compute_propability_density_histogram(std::vector<double>& bucket_density, double count)
	{
		for (int i = 0; i < m_histogram.binCount; i++)
		{
			double next_edge = m_histogram.min + (m_histogram.binSize * (static_cast<double>(i) + 1.0));
			double curr_edge = m_histogram.min + (m_histogram.binSize * static_cast<double>(i));
			double diff = next_edge - curr_edge;
			bucket_density[i] = m_bucketFrequency[i] / diff / count;
		}
	}
}

