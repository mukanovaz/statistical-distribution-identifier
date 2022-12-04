#pragma once
#include <cmath>
#include <vector>
#include <memory>
#include "rss.cpp"
#include "data.h"

#undef min
#undef max

#include <tbb/parallel_for.h>


namespace ppr::hist
{
	/// <summary>
	/// Class is using to create data histogram. (Using only for sequential computing)
	/// </summary>
	class Histogram
	{
	private:
		double BinSize;
		double Max, Min, Size, ScaleFactor;

	public:
		Histogram(int size, double bin_size, double min, double max);

		/// <summary>
		/// Process one number and update histogram
		/// </summary>
		/// <param name="arr">- Histogram vector reference</param>
		/// <param name="x">- One number</param>
		void push(std::vector<int>& arr, double x);

		/// <summary>
		/// Transform frequency histogram to propability density histogram
		/// </summary>
		/// <param name="hist">- Histogram configuration structure</param>
		/// <param name="bucket_frequency">- Frequency histogram reference</param>
		/// <param name="bucket_density">- Density histogram reference</param>
		/// <param name="count">- All data count</param>
		void compute_propability_density_histogram(std::vector<double>& bucket_density, std::vector<int>& bucket_frequency, double count);

		/// <summary>
		/// Compute RSS of histogram
		/// </summary>
		/// <param name="bucket_density">- Density histogram reference</param>
		/// <param name="dist_val">- Distribution</param>
		/// <param name="res">- Final results</param>
		/// <returns>RSS value</returns>
		double compute_rss_histogram(std::vector<double>& bucket_density, char dist_val, SResult& res);
	};

	class Histogram_parallel
	{
	private:
		SHistogram m_histogram;
		const double* m_data;
		double m_mean;

	public:
		double m_var;
		std::vector<int> m_bucketFrequency;

		Histogram_parallel(int size, double bin_size, double min, double max, const double* data, double mean);

		Histogram_parallel(Histogram_parallel& x, tbb::split);

		/// <summary>
		/// TBB operator() method
		/// </summary>
		/// <param name="r">- blocked_range</param>
		void operator()(const tbb::blocked_range<size_t>& r);

		/// <summary>
		/// TBB join() method
		/// </summary>
		/// <param name="y">- Another instance of class to reduce</param>
		void join(const Histogram_parallel& y);

		/// <summary>
		/// Transform frequency histogram to propability density histogram
		/// </summary>
		/// <param name="hist">- Histogram configuration structure</param>
		/// <param name="bucket_frequency">- Frequency histogram reference</param>
		/// <param name="bucket_density">- Density histogram reference</param>
		/// <param name="count">- All data count</param>
		void compute_propability_density_histogram(std::vector<double>& bucket_density, double count);
	};


}
