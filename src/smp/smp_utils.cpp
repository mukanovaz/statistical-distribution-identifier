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
		SDataStat local_stat;
		cl_int err = 0;
		const unsigned long long work_group_number = data_count / m_ocl_config.wg_size;
		const unsigned int count = data_count - (data_count % m_ocl_config.wg_size);
		cl::CommandQueue cmd_queue(m_ocl_config.context, m_ocl_config.device, 0, &err);

		cl::Buffer in_data_buf(m_ocl_config.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_USE_HOST_PTR, count * sizeof(double), data, &err);
		cl::Buffer out_sum_buf(m_ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
		cl::Buffer out_min_buf(m_ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);
		cl::Buffer out_max_buf(m_ocl_config.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, work_group_number * sizeof(double), nullptr, &err);

		// Set method arguments
		err = m_ocl_config.kernel.setArg(0, in_data_buf);
		err = m_ocl_config.kernel.setArg(1, m_ocl_config.wg_size * sizeof(double), nullptr);
		err = m_ocl_config.kernel.setArg(2, m_ocl_config.wg_size * sizeof(double), nullptr);
		err = m_ocl_config.kernel.setArg(3, m_ocl_config.wg_size * sizeof(double), nullptr);
		err = m_ocl_config.kernel.setArg(4, out_sum_buf);
		err = m_ocl_config.kernel.setArg(5, out_min_buf);
		err = m_ocl_config.kernel.setArg(6, out_max_buf);

		// Result data
		std::vector<double> out_sum(work_group_number);
		std::vector<double> out_min(work_group_number);
		std::vector<double> out_max(work_group_number);


		// Pass all data to GPU
		err = m_ocl_config.queue.enqueueNDRangeKernel(m_ocl_config.kernel, cl::NullRange, cl::NDRange(count), cl::NDRange(m_ocl_config.wg_size));
		err = m_ocl_config.queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_sum.data());
		err = m_ocl_config.queue.enqueueReadBuffer(out_min_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_min.data());
		err = m_ocl_config.queue.enqueueReadBuffer(out_max_buf, CL_TRUE, 0, work_group_number * sizeof(double), out_max.data());

		cl::finish();

		// Agregate results on CPU
		double sum = ppr::parallel::sum_vector_elements_vectorized(out_sum);
		double max = ppr::parallel::max_of_vector_vectorized(out_max);
		double min = ppr::parallel::min_of_vector_vectorized(out_min);

		return {
			count,					// n
			sum,					// sum
			max,					// max
			min,					// min
			0.0,					// mean
			0.0,					// variance
			true
		};

		return local_stat;
	}

	std::tuple<std::vector<int>, double> CHistProcessingUnit::RunCPU(double* data, int data_count)
	{
		std::vector<int> local_vector(m_hist.binCount + 1);
		double variance = 0.0;

		GetHistogramVectorized(local_vector, variance, data_count, data, m_hist, m_stat);
		
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

		double* max_d = (double*)&max;
		return std::max({ max_d[0], max_d[1], max_d[2], max_d[3] });
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

		double* min_d = (double*)&min;
		return std::min({ min_d[0], min_d[1], min_d[2], min_d[3] });
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
