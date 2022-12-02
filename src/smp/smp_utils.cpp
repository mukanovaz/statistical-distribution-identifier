#include "smp_utils.h"
#include "../opencl/gpu_utils.h"
#include "../executor.h"

namespace ppr::parallel
{
	SDataStat CStatProcessingUnit::RunCPU(double* data, int data_count)
	{
		SDataStat local_stat;

		GetStatisticsVectorized(local_stat, data_count, data);

		return local_stat;
	}

	SDataStat CStatProcessingUnit::RunGPU(double* data, int data_count)
	{
		return ppr::gpu::RunStatisticsOnGPU(m_ocl_config, m_configuration, data, data_count);
	}

	std::tuple<std::vector<int>, double> CHistProcessingUnit::RunCPU(double* data, int data_count)
	{
		std::vector<int> local_vector(m_hist.binCount + 1);
		double variance = 0.0;

		GetHistogramVectorized(local_vector, variance, data_count, data, m_hist, m_stat);

		return std::make_tuple(local_vector, variance);
	}

	std::tuple<std::vector<int>, double> CHistProcessingUnit::RunGPU(double* data, int data_count)
	{
		std::vector<int> local_vector(m_hist.binCount + 1);
		double variance = 0.0;

		ppr::gpu::RunHistogramOnGPU(m_ocl_config, m_configuration, m_hist, m_stat, data, data_count, local_vector, variance);

		return std::make_tuple(local_vector, variance);
	}

	void GetHistogramVectorized(std::vector<int>& local_vector, double& variance, int data_count, double* data, SHistogram& hist, SDataStat& stat)
	{
		// Fill vector with mean/min and scale value
		const __m256d mean = _mm256_set1_pd(stat.mean);
		const __m256d min = _mm256_set1_pd(stat.min);
		const __m256d scale = _mm256_set1_pd(hist.scaleFactor);

		for (int block = 0; block < data_count; block += 4)
		{
			__m256d vec = _mm256_load_pd(data + block);
			// Compute variance of 4 vector elements
			variance += variance_double_avx(vec, mean);

			// Find position for 4 elements
			__m256d position = position_double_avx(vec, min, scale);
			double* pos = (double*)&position;
			local_vector[pos[0]] += 1;
			local_vector[pos[1]] += 1;
			local_vector[pos[2]] += 1;
			local_vector[pos[3]] += 1;
		}
	}

	double max_of_vector_vectorized(std::vector<double> vector)
	{
		__m256d max = _mm256_set1_pd(
			std::numeric_limits<double>::min()
		);

		int size = vector.size() - (vector.size() % 4);
		for (int block = 0; block < size; block += 4)
		{
			__m256d vec = _mm256_set_pd(
				vector[block],
				vector[block + 1],
				vector[block + 2],
				vector[block + 3]
			);
			max = _mm256_max_pd(max, vec);
		}

		// Find min on the rest of the vector
		double* max_d = (double*)&max;
		if (vector.size() - size != 0)
		{
			double max_l = std::numeric_limits<double>::min();
			int size2 = vector.size() - size;
			for (int i = 0; i < size2; i++)
			{
				max_l = std::max({ max_l, vector[size + i] });
			}
			return std::max({ max_d[0], max_d[1], max_d[2], max_d[3], max_l });
		}
		else
		{
			return std::max({ max_d[0], max_d[1], max_d[2], max_d[3] });
		}
	}

	double min_of_vector_vectorized(std::vector<double> vector)
	{
		__m256d min = _mm256_set1_pd(
			std::numeric_limits<double>::max()
		);

		int size = vector.size() - (vector.size() % 4);
		for (int block = 0; block < size; block += 4)
		{
			__m256d vec = _mm256_set_pd(
				vector[block],
				vector[block + 1],
				vector[block + 2],
				vector[block + 3]
			);
			min = _mm256_min_pd(min, vec);
		}

		// Find min on the rest of the vector
		double* min_d = (double*)&min;
		if (vector.size() - size != 0)
		{
			double min_l = std::numeric_limits<double>::max();
			int size2 = vector.size() - size;
			for (int i = 0; i < size2; i++)
			{
				min_l = std::min({min_l, vector[size + i] });
			}
			return std::min({ min_d[0], min_d[1], min_d[2], min_d[3], min_l });
		}
		else
		{
			return std::min({ min_d[0], min_d[1], min_d[2], min_d[3] });
		}
		
	}

	double sum_vector_elements_vectorized(std::vector<double> vector)
	{
		double sum = 0.0;
		int size = vector.size() - (vector.size() % 4);
		for (int block = 0; block < size; block += 4)
		{
			__m256d vec = _mm256_set_pd(
				vector[block],
				vector[block + 1],
				vector[block + 2],
				vector[block + 3]
			);
			sum += hsum_double_avx(vec);
		}

		// Sum the rest if exist
		if (vector.size() - size != 0)
		{
			int size2 = vector.size() - size;
			for (int i = 0; i < size2; i++)
			{
				sum += vector[size + i];
			}
		}

		return sum;
	}

	void GetStatisticsVectorized(SDataStat& stat, unsigned int data_count, double* data)
	{
		__m256d min = _mm256_set1_pd(
			std::numeric_limits<double>::max()
		);

		__m256d max = _mm256_set1_pd(
			std::numeric_limits<double>::lowest()
		);

		for (int block = 0; block < data_count; block += 4)
		{
			// Count elements
			stat.n += 4;

			// Find sum of 4 vector element
			__m256d vec = _mm256_load_pd(data + block);
			stat.sum += hsum_double_avx(vec);

			// Find Max and Min of 4 vector element
			max = _mm256_max_pd(max, vec);
			min = _mm256_min_pd(min, vec);
		}

		// Agregate results
		double* min_d = (double*)& min;
		double* max_d = (double*)& max;
		stat.min = std::min({ min_d[0], min_d[1], min_d[2], min_d[3] });
		stat.max = std::max({ max_d[0], max_d[1], max_d[2], max_d[3] });
	}

	inline __m256d position_double_avx(__m256d v, __m256d min, __m256d scale) {
		__m256d sub = _mm256_sub_pd(v, min);
		return _mm256_mul_pd(sub, scale);
	}

	inline double variance_double_avx(__m256d v, __m256d mean) {
		__m256d sub = _mm256_sub_pd(v, mean);
		__m256d mul = _mm256_mul_pd(sub, sub);

		return  hsum_double_avx(mul);			// reduce to scalar
	}

	inline double hsum_double_avx(__m256d v) {						// s1 = 1 2 3 4 5 6 7 8
		__m128d vlow = _mm256_castpd256_pd128(v);					// l1 = 1 2 3 4
		__m128d vhigh = _mm256_extractf128_pd(v, 1);				// h1 = 5 6 7 8
		vlow = _mm_add_pd(vlow, vhigh);								// s2 = 6 8 10 12

		__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
		return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));			// reduce to scalar
	}
}
