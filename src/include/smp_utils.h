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
		/// <returns></returns>
		SDataStat run_on_CPU();

		/// <summary>
		/// Collect statistics of data block using OpenCL device. (Not using)
		/// </summary>
		/// <returns></returns>
		SDataStat run_on_GPU();
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
		/// <returns>Histogram vector and variance</returns>
		std::tuple<std::vector<int>, double> run_on_CPU();

		/// <summary>
		/// Create frequency histogram of data block using AVX2 instructions.
		/// </summary>
		/// <returns>Histogram vector and variance</returns>
		std::tuple<std::vector<int>, double> run_on_GPU();
	};

	void agregate_gpu_stat_vectorized(SDataStat& stat, double* array_sum, double* array_min, double* array_max, int size);

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
	/// <param name="variance">- Variance</param>
	/// <param name="data_count">- Data count</param>
	/// <param name="data">- Data pointer</param>
	/// <param name="hist">- Histogram configration structure</param>
	/// <param name="stat">- Statistics structure</param>
	/// <returns>- Histogram vector</returns>
	void get_histogram(double& variance, long long data_count, double* data, SHistogram& hist, SDataStat& stat, std::vector<int>& histogram);

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