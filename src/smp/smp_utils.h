#pragma once
#include "../config.h"
#include "../data.h"
#include <vector>

#undef min
#undef max

namespace ppr::parallel
{
	class CStatProcessingUnit
	{
	private:
		SConfig m_configuration;
		SOpenCLConfig m_ocl_config;

	public:
		CStatProcessingUnit(SConfig& config, SOpenCLConfig& ocl_config) : m_configuration(config), m_ocl_config(ocl_config) {}

		SDataStat RunCPU(double* data, int data_count);

		SDataStat RunGPU(double* data, 
			std::vector<double>& out_sum,
			std::vector<double>& out_min,
			std::vector<double>& out_max);
	};

	class CHistProcessingUnit
	{
	private:
		SConfig m_configuration;
		SOpenCLConfig m_ocl_config;
		SHistogram m_hist;
		SDataStat m_stat;

	public:
		CHistProcessingUnit(SHistogram& hist, SConfig& config, SOpenCLConfig& ocl_config, SDataStat& stat) 
			: m_configuration(config), m_ocl_config(ocl_config), m_hist(hist), m_stat(stat) {}

		std::tuple<std::vector<int>, double> RunCPU(double* data, int data_count);
	};

	inline __m256d position_double_avx(__m256d v, __m256d min, __m256d scale);

	inline double variance_double_avx(__m256d v, __m256d mean);

	// https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
	inline double hsum_double_avx(__m256d v);


	void GetStatisticsVectorized(SDataStat& stat, unsigned int data_count, double* data);

	void GetHistogramVectorized(std::vector<int>& local_vector, double& variance, int data_count, double* data, SHistogram& hist, SDataStat& stat);
}