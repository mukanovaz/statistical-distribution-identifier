#pragma once
#include "config.h"
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
		SOpenCLConfig m_ocl_config;

	public:
		Stat_processing_unit(SConfig& config, SOpenCLConfig& ocl_config) : m_configuration(config), m_ocl_config(ocl_config) {}

		/// <summary>
		/// Collect statistics of data block using AVX2 instructions
		/// </summary>
		/// <param name="data">data block pointer</param>
		/// <param name="data_count">data count</param>
		/// <returns></returns>
		SDataStat run_on_CPU(double* data, int data_count);

		/// <summary>
		/// Collect statistics of data block using OpenCL device. (Not using)
		/// </summary>
		/// <param name="data">data block pointer</param>
		/// <param name="data_count">data count</param>
		/// <returns></returns>
		SDataStat run_on_GPU(double* data, int data_count);
	};

	class Hist_processing_unit
	{
	private:
		SConfig m_configuration;
		SOpenCLConfig m_ocl_config;
		SHistogram m_hist;
		SDataStat m_stat;

	public:
		Hist_processing_unit(SHistogram& hist, SConfig& config, SOpenCLConfig& ocl_config, SDataStat& stat) 
			: m_configuration(config), m_ocl_config(ocl_config), m_hist(hist), m_stat(stat) {}

		/// <summary>
		/// Create frequency histogram of data block using OpenCL device. (Not using)
		/// </summary>
		/// <param name="data">- data block pointer</param>
		/// <param name="data_count">- data count</param>
		/// <returns>Histogram vector and variance</returns>
		std::tuple<std::vector<int>, double> run_on_CPU(double* data, int data_count);

		/// <summary>
		/// Create frequency histogram of data block using AVX2 instructions.
		/// </summary>
		/// <param name="data">- data block pointer</param>
		/// <param name="data_count">- data count</param>
		/// <returns>Histogram vector and variance</returns>
		std::tuple<std::vector<int>, double> run_on_GPU(double* data, int data_count);
	};


	/// <summary>
	/// Main function to start collecting statistics from input data using AVX2 instructions.
	/// </summary>
	/// <param name="stat">- Statistics structure</param>
	/// <param name="data_count">- Data count</param>
	/// <param name="data">- Data pointer</param>
	void get_statistics_vectorized(SDataStat& stat, unsigned int data_count, double* data);

	/// <summary>
	/// Main function to start creating frequency histogram from input data using AVX2 instructions. (Not using)
	/// </summary>
	/// <param name="local_vector">- Histogram vector reference</param>
	/// <param name="variance">- Variance</param>
	/// <param name="data_count">- Data count</param>
	/// <param name="data">- Data pointer</param>
	/// <param name="hist">- Histogram configration structure</param>
	/// <param name="stat">- Statistics structure</param>
	void get_histogram_vectorized(std::vector<int>& local_vector, double& variance, int data_count, double* data, SHistogram& hist, SDataStat& stat);

	/// <summary>
	/// Find a minimum value of std::vector using AVX2 instructions.
	/// </summary>
	/// <param name="vector">- Vector with data (count should be multipy of 4)</param>
	/// <returns>Minimum value</returns>
	double min_of_vector_vectorized(std::vector<double> vector);

	/// <summary>
	/// Find a maximum value of std::vector using AVX2 instructions.
	/// </summary>
	/// <param name="vector">- Vector with data (count should be multipy of 4)</param>
	/// <returns>Maximum value</returns>
	double max_of_vector_vectorized(std::vector<double> vector);

	/// <summary>
	/// Computes sum of all vector elements using AVX2 instructions.
	/// </summary>
	/// <param name="vector">- Vector with data (count should be multipy of 4)</param>
	/// <returns>Sum of all vector elements</returns>
	double sum_vector_elements_vectorized(std::vector<double> vector);

	/// <summary>
	/// Computing number position in histogram using AVX2 instructions.
	/// </summary>
	/// <param name="v">- 4 element vector with data in double type</param>
	/// <param name="min">- 4 element vector with minimum element of all data set (constant)</param>
	/// <param name="scale">- 4 element vector with histogram scale (constant)</param>
	/// <returns>position in histogram</returns>
	inline __m256d position_double_avx(__m256d v, __m256d min, __m256d scale);

	/// <summary>
	/// Computing data variance using AVX2 instructions.
	/// </summary>
	/// <param name="v">- 4 element vector with data in double type</param>
	/// <param name="mean">- 4 element vector mean value of all data set (constant)</param>
	/// <returns>variance</returns>
	inline double variance_double_avx(__m256d v, __m256d mean);

	/// <summary>
	/// Computes sum of 4 element vector using AVX2 instructions.
	/// https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
	/// </summary>
	/// <param name="v">- 4 element vector with data in double type</param>
	/// <returns>sum of vector elements</returns>
	inline double hsum_double_avx(__m256d v);
}