#pragma once
#include "config.h"
#include "gpu_utils.h"
#include "data.h"
#include <vector>

#undef min
#undef max

namespace ppr::parallel
{
	/// <summary>
	/// Class for getting statistics. Is using for one thread processing
	/// </summary>
	class Stat_processing_unit
	{
	private:
		SConfig m_configuration;
		ppr::gpu::SOpenCLConfig m_ocl_config;

	public:
		Stat_processing_unit(SConfig& config, ppr::gpu::SOpenCLConfig& ocl_config) : m_configuration(config), m_ocl_config(ocl_config) {}

		/// <summary>
		/// Collect statistics of data block using AVX2 instructions
		/// </summary>
		/// <param name="data">data block pointer</param>
		/// <param name="data_count">data count</param>
		/// <returns></returns>
		SDataStat run_on_CPU(double* data, long long data_count);

		/// <summary>
		/// Collect statistics of data block using OpenCL device. (Not using)
		/// </summary>
		/// <param name="data">data block pointer</param>
		/// <param name="data_count">data count</param>
		/// <returns></returns>
		SDataStat run_on_GPU(double* data, long long begin, long long end);
	};

	class Hist_processing_unit
	{
	private:
		SConfig m_configuration;
		ppr::gpu::SOpenCLConfig m_ocl_config;
		SHistogram m_hist;
		SDataStat m_stat;

	public:
		Hist_processing_unit(SHistogram& hist, SConfig& config, ppr::gpu::SOpenCLConfig& ocl_config, SDataStat& stat)
			: m_configuration(config), m_ocl_config(ocl_config), m_hist(hist), m_stat(stat) {}

		/// <summary>
		/// Create frequency histogram of data block using OpenCL device. (Not using)
		/// </summary>
		/// <param name="data">- data block pointer</param>
		/// <param name="data_count">- data count</param>
		/// <returns>Histogram vector and variance</returns>
		std::tuple<std::vector<int>, double> run_on_CPU(double* data, long long data_count);

		/// <summary>
		/// Create frequency histogram of data block using AVX2 instructions.
		/// </summary>
		/// <param name="data">- data block pointer</param>
		/// <param name="data_count">- data count</param>
		/// <returns>Histogram vector and variance</returns>
		std::tuple<std::vector<int>, double> run_on_GPU(double* data, long long begin, long long end);
	};

	/// <summary>
	/// Main function to start creating frequency histogram from input data using AVX2 instructions. (Not using)
	/// </summary>
	/// <param name="stat">- Statistics structure</param>
	/// <param name="array_sum">- input array</param>
	/// <param name="array_min">- input array</param>
	/// <param name="array_max">- input array</param>
	/// <param name="size">- array size</param>
	void agregate_gpu_stat_vectorized(SDataStat& stat, double* array_sum, double* array_min, double* array_max, int size);
	
	/// <summary>
	/// Find sum of all elements of array. Vectorized
	/// </summary>
	/// <param name="array">- input array</param>
	/// <param name="size">- array size</param>
	/// <returns></returns>
	double sum_vector_elements_vectorized(double* array, int size);

	/// <summary>
	/// Main function to start collecting statistics from input data using AVX2 instructions.
	/// </summary>
	/// <param name="stat">- Statistics structure</param>
	/// <param name="data_count">- Data count</param>
	/// <param name="data">- Data pointer</param>
	void get_statistics_vectorized(SDataStat& stat, long long data_count, double* data);

	/// <summary>
	/// Main function to start creating frequency histogram from input data using AVX2 instructions. (Not using)
	/// </summary>
	/// <param name="local_vector">- Histogram vector reference</param>
	/// <param name="variance">- Variance</param>
	/// <param name="data_count">- Data count</param>
	/// <param name="data">- Data pointer</param>
	/// <param name="hist">- Histogram configration structure</param>
	/// <param name="stat">- Statistics structure</param>
	void get_histogram_vectorized(std::vector<int>& local_vector, double& variance, long long data_count, double* data, SHistogram& hist, SDataStat& stat);

	/// <summary>
	/// Find a maximum value of std::vector using AVX2 instructions.
	/// </summary>
	/// <param name="vector">- Vector with data (count should be multipy of 4)</param>
	/// <returns>Maximum value</returns>
	double max_of_vector_vectorized(std::vector<double> vector);

	/// <summary>
	/// Calculating RSS for each distribution i threads
	/// </summary>
	/// <param name="res">- Final results structure</param>
	/// <param name="histogramDensity">- Density histogram reference</param>
	/// <param name="hist">- Histogram configuration structure</param>
	void calculate_histogram_RSS_cpu(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist);

	/// <summary>
	/// Method to calculate RSS for Gauss distribution. Used in thread
	/// </summary>
	/// <param name="res">- Final results structure</param>
	/// <param name="histogramDensity">- Density histogram reference</param>
	/// <param name="hist">- Histogram configuration structure</param>
	/// <returns>rss</returns>
	double get_gauss_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist);

	/// <summary>
	/// Method to calculate RSS for Gauss distribution. Used in thread
	/// </summary>
	/// <param name="res">- Final results structure</param>
	/// <param name="histogramDensity">- Density histogram reference</param>
	/// <param name="hist">- Histogram configuration structure</param>
	/// <returns>rss</returns>
	double get_poisson_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist);

	/// <summary>
	/// Method to calculate RSS for Exponential distribution. Used in thread
	/// </summary>
	/// <param name="res">- Final results structure</param>
	/// <param name="histogramDensity">- Density histogram reference</param>
	/// <param name="hist">- Histogram configuration structure</param>
	/// <returns>rss</returns>
	double get_exp_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist);

	/// <summary>
	/// Method to calculate RSS for Uniform distribution. Used in thread
	/// </summary>
	/// <param name="res">- Final results structure</param>
	/// <param name="histogramDensity">- Density histogram reference</param>
	/// <param name="hist">- Histogram configuration structure</param>
	/// <returns>rss</returns>
	double get_uniform_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist);
}