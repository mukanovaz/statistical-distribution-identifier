#include "smp_solver.h"

namespace ppr::parallel
{
	SResult run(SConfig& configuration)
	{
		//  ================ [Init TBB]
		tbb::task_arena arena(configuration.thread_count == 0 ? tbb::task_arena::automatic : static_cast<int>(configuration.thread_count));

		//  ================ [Map input file]
		FileMapping mapping(configuration.input_fn);

		//  ================ [Allocations]
		SOpenCLConfig opencl; // Not using here
		SHistogram hist;
		SResult res;
		SDataStat stat;
		std::vector<int> tmp(0);
		unsigned int data_count = mapping.GetCount();

		//  ================ [Get statistics]
		tbb::tick_count t0 = tbb::tick_count::now();
		mapping.ReadInChunks(hist, configuration, opencl, stat, arena, tmp, &GetStatisticsCPU);
		tbb::tick_count t1 = tbb::tick_count::now();
		std::cout << "Statistics:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		// Find mean
		stat.mean = stat.sum / stat.n;

		//  ================ [Create frequency histogram]
		// Update kernel program
		ppr::gpu::UpdateProgram(opencl, HIST_KERNEL, HIST_KERNEL_NAME);

		// Find histogram limits
		hist.binCount = log2(stat.n) + 1;
		hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		std::vector<int> histogramFreq(static_cast<int>(hist.binCount));
		std::vector<double> histogramDensity(static_cast<int>(hist.binCount));
		cl_int err = 0;

		// Run
		t0 = tbb::tick_count::now();
		mapping.ReadInChunks(hist, configuration, opencl, stat, arena, histogramFreq, &CreateFrequencyHistogramCPU);
		t1 = tbb::tick_count::now();
		std::cout << "Histogram:\t" << (t1 - t0).seconds() << "\tsec." << std::endl;

		// Find variance
		stat.variance = stat.variance / stat.n;

		//  ================ [Create density histogram]
		ppr::executor::ComputePropabilityDensityOfHistogram(hist, histogramFreq, histogramDensity, stat.n);

		//  ================ [Fit params]
		res.isNegative = stat.isNegative;
		res.isInteger = std::floor(stat.sum) == stat.sum;

		// Gauss maximum likelihood estimators
		res.gauss_mean = stat.mean;
		res.gauss_variance = stat.variance;
		res.gauss_stdev = sqrt(stat.variance);

		// Exponential maximum likelihood estimators
		res.exp_lambda = stat.n / stat.sum;

		// Poisson likelihood estimators
		res.poisson_lambda = stat.sum / stat.n;

		// Uniform likelihood estimators
		res.uniform_a = stat.min;
		res.uniform_b = stat.max;

		//	================ [Calculate RSS]
		ppr::executor::CalculateHistogramRSSOnCPU(res, arena, histogramDensity, hist);

		//	================ [Analyze]
		ppr::executor::AnalyzeResults(res);

		std::cout << "Finish." << std::endl;
		return res;
	}

	void avx2_sum_64_block(SDataStat& stat, double* vec_first, double* vec_second, __m256d& max, __m256d& min) noexcept
	{
		__m256d accumulator1 = _mm256_setzero_pd();
		__m256d accumulator2 = _mm256_setzero_pd();

		for (std::size_t block = 0; block < 64; block += 4 * 2)
		{
			__m256d vec_f1 = _mm256_loadu_pd(vec_first + block);
			__m256d vec_f2 = _mm256_loadu_pd(vec_first + block + 4);
			__m256d vec_s1 = _mm256_loadu_pd(vec_second + block);
			__m256d vec_s2 = _mm256_loadu_pd(vec_second + block + 4);

			__m256d tmp1 = _mm256_add_pd(vec_f1, vec_s1);
			__m256d tmp2 = _mm256_add_pd(vec_f2, vec_s2);
			accumulator1 = _mm256_add_pd(accumulator1, tmp1);
			accumulator2 = _mm256_add_pd(accumulator2, tmp2);

			max = _mm256_max_pd(max, vec_f1);
			max = _mm256_max_pd(max, vec_f2);
			max = _mm256_max_pd(max, vec_s1);
			max = _mm256_max_pd(max, vec_s2);

			min = _mm256_min_pd(min, vec_f1);
			min = _mm256_min_pd(min, vec_f2);
			min = _mm256_min_pd(min, vec_s1);
			min = _mm256_min_pd(min, vec_s2);
		}

		accumulator1 = _mm256_add_pd(accumulator1, accumulator2);

		__m128d accumulator = _mm_add_pd(_mm256_castpd256_pd128(accumulator1),
			_mm256_extractf128_pd(accumulator1, 1));
		accumulator = _mm_add_pd(accumulator,
			_mm_unpackhi_pd(accumulator, accumulator));

		stat.sum += _mm_cvtsd_f64(accumulator);
	}


	// https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
	inline
		double hsum_double_avx(__m256d v) {
		__m128d vlow = _mm256_castpd256_pd128(v);
		__m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
		vlow = _mm_add_pd(vlow, vhigh);     // reduce down to 128

		__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
		return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
	}


	// https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
	void avx2_sum_64_block2(double& sum, double* vec_first, double* vec_second) noexcept
	{
		__m256d accumulator1 = _mm256_setzero_pd();
		__m256d accumulator2 = _mm256_setzero_pd();

		for (std::size_t block = 0; block < 64; block += 4)
		{
			__m256d vec_f1 = _mm256_loadu_pd(vec_first + block);
			__m256d vec_s1 = _mm256_loadu_pd(vec_second + block);

			double s1 = hsum_double_avx(vec_f1);
			double s2 = hsum_double_avx(vec_s1);

			sum += s1 + s2;
		}
	}

	void GetStatisticsVectorized(SHistogram& hist, SConfig& configuration, SDataStat& stat, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		int half_count = data_count / 2;
		__m256d min = _mm256_set_pd(
			std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max(),
			std::numeric_limits<double>::max()
		);

		__m256d max = _mm256_set_pd(
			std::numeric_limits<double>::lowest(),
			std::numeric_limits<double>::lowest(),
			std::numeric_limits<double>::lowest(),
			std::numeric_limits<double>::lowest()
		);

 		for (int block = 0; block < half_count; block += 64)
		{
			avx2_sum_64_block(stat, data + block, data + block + half_count, max, min);
		}
		double test = 0.0;
	}

	void GetStatisticsCPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		if (USE_VECTORIZATION)
		{
			tbb::tick_count t0 = tbb::tick_count::now();
			GetStatisticsVectorized(hist, configuration, stat, data_count, data, histogram);
			tbb::tick_count t1 = tbb::tick_count::now();
			std::cout << (t1 - t0).seconds() << std::endl;
		}
		else
		{
			// Find rest of a statistics on CPU
			RunningStatParallel stat_cpu(data, opencl.data_count_for_cpu);
			ppr::executor::RunOnCPU<RunningStatParallel>(arena, stat_cpu, opencl.data_count_for_cpu + 1, data_count);

			// Agregate results results
			stat.n += stat_cpu.NumDataValues();
			stat.min = std::min({ stat.min, std::min({ stat.min, stat_cpu.Get_Min() }) });
			stat.max = std::max({ stat.max, std::max({ stat.max, stat_cpu.Get_Max() }) });
			stat.sum += stat_cpu.Sum();
			stat.isNegative = stat.isNegative || stat_cpu.IsNegative();
		}
	}

	void CreateFrequencyHistogramCPU(SHistogram& hist, SConfig& configuration, SOpenCLConfig& opencl, SDataStat& stat, tbb::task_arena& arena, unsigned int data_count, double* data, std::vector<int>& histogram)
	{
		// Run on CPU
		ppr::hist::HistogramParallel hist_cpu(hist.binCount, hist.binSize, stat.min, stat.max, data, stat.mean);
		ppr::executor::RunOnCPU<ppr::hist::HistogramParallel>(arena, hist_cpu, opencl.data_count_for_cpu, data_count);

		// Transform vector
		std::transform(histogram.begin(), histogram.end(), hist_cpu.m_bucketFrequency.begin(), histogram.begin(), std::plus<int>());

		stat.variance += hist_cpu.m_var; // GPU + CPU "half" variance
	}


}