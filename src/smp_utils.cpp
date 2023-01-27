#include "include/smp_utils.h"
#include "include/gpu_utils.h"
#include "include/executor.h"
#include "include/file_mapper.h"
#include <future>
#include <execution>


namespace ppr::parallel
{
	SDataStat Stat_processing_unit::run_on_CPU()
	{
		// Local variables
		SDataStat local_stat;
		File_mapper* mapper = File_mapper::get_instance();
		long long data_count = m_ocl_config.data_count / sizeof(double);

		double* pView = mapper->view(m_ocl_config.high, m_ocl_config.low, m_ocl_config.data_count);

		if (pView == NULL)
		{
			return local_stat;
		}

		get_statistics_vectorized(local_stat, data_count, pView);

		UnmapViewOfFile(pView);

		return local_stat;
	}

	SDataStat Stat_processing_unit::run_on_GPU()
	{
		SDataStat local_stat;
		File_mapper* mapper = File_mapper::get_instance();
		
		long long data_count = m_ocl_config.data_count / sizeof(double);

		// Map a file
		double* pView = mapper->view(m_ocl_config.high, m_ocl_config.low, m_ocl_config.data_count);

		if (pView == NULL)
		{
			return local_stat;
		}

		// Run
		ppr::gpu::run_statistics_on_device(m_ocl_config, local_stat, data_count, pView);

		// Unmap
		UnmapViewOfFile(pView);

		return local_stat;
	}

	std::tuple<std::vector<int>, double> Hist_processing_unit::run_on_CPU()
	{
		// Local variables
		double variance = 0.0;
		File_mapper* mapper = File_mapper::get_instance();
		long long data_count = m_ocl_config.data_count / sizeof(double);
		std::vector<int> local_vector(m_hist.binCount + 1);

		double* pView = mapper->view(m_ocl_config.high, m_ocl_config.low, m_ocl_config.data_count);

		if (pView == NULL)
		{
			return std::make_tuple(local_vector, variance);
		}

		get_histogram(variance, data_count, pView, m_hist, m_stat, local_vector);

		UnmapViewOfFile(pView);

		// Create return value
		return std::make_tuple(local_vector, variance);
	}

	std::tuple<std::vector<int>, double> Hist_processing_unit::run_on_GPU()
	{
		// Local variables
		std::vector<int> local_vector(m_hist.binCount + 1);
		double variance = 0.0;
		File_mapper* mapper = File_mapper::get_instance();
		long long data_count = m_ocl_config.data_count / sizeof(double);

		double* pView = mapper->view(m_ocl_config.high, m_ocl_config.low, m_ocl_config.data_count);

		if (pView == NULL)
		{
			return std::make_tuple(local_vector, variance);
		}

		// Call Opencl kernel
		ppr::gpu::run_histogram_on_device(m_ocl_config, m_stat, data_count, pView, m_hist, local_vector, variance);

		UnmapViewOfFile(pView);

		// Create return value
		return std::make_tuple(local_vector, variance);
	}

	void get_histogram(double& variance, long long data_count, double* data, SHistogram& hist, SDataStat& stat, std::vector<int>& histogram)
	{
		double mean = stat.mean;
		double min = stat.min;
		double scale = hist.scaleFactor;
		double variance_local = 0.0;
		
		for (int i = 0; i < data_count; i++)
		{
			// Update histogram
			int x = (int)data[i];
		 	double position = (data[i] - min) * scale;
			histogram[static_cast<int>(position)] += 1;

			// Find variance
			double tmp = data[i] - mean;
			variance_local = variance_local + (tmp * tmp);
		}

		variance = variance_local;
	}

	void get_statistics_vectorized(SDataStat& stat, long long data_count, double* data)
	{
		int half_count = data_count / 2;
		double* data_right = data;
		double* data_left = data + half_count;

		std::vector<double> sum(half_count);
		std::vector<double> min(half_count);
		std::vector<double> max(half_count);

		int count = data_count;

		for (int i = 0; i < half_count - 1; ++i)
		{
			sum[i] = data_right[i] + data_left[i];
			min[i] = data_right[i] < data_left[i] ? data_right[i] : data_left[i];
			max[i] = data_right[i] > data_left[i] ? data_right[i] : data_left[i];
		}

		stat.sum = std::reduce(std::execution::par, sum.cbegin(), sum.cend());
		stat.n = data_count;
		const auto [min_val, xxx1] = std::minmax_element(begin(min), end(min));
		const auto [xxx2, max_val] = std::minmax_element(begin(max), end(max));
		stat.min = *min_val;
		stat.max = *max_val;
	}

	void agregate_gpu_stat_vectorized(SDataStat& stat, double* array_sum, double* array_min, double* array_max, int size)
	{
		int half_count = size / 2;

		double* array_sum_left = array_sum + half_count;
		double* array_min_left = array_min + half_count;
		double* array_max_left = array_max + half_count;

		std::vector<double> sum(half_count);
		std::vector<double> min(half_count);
		std::vector<double> max(half_count);

		for (int i = 0; i < half_count - 1; i++)
		{
			sum[i] = array_sum[i] + array_sum_left[i];
			min[i] = array_min[i] < array_min_left[i] ? array_min[i] : array_min_left[i];
			max[i] = array_max[i] > array_max_left[i] ? array_max[i] : array_max_left[i];
		}

		stat.sum = std::reduce(std::execution::par_unseq, sum.cbegin(), sum.cend());
		const auto [min_val, xxx1] = std::minmax_element(begin(min), end(min));
		const auto [xxx2, max_val] = std::minmax_element(begin(max), end(max));
		stat.min = *min_val;
		stat.max = *max_val;
	}

	double sum_vector_elements_vectorized(double* data, int size)
	{
		int half_count = size / 2;
		double* data_right = data;
		double* data_left = data + half_count;

		std::vector<double> sum(half_count);
		
		for (int i = 0; i < half_count - 1; i++)
		{
			sum[i] = data_right[i] + data_left[i];
		}

		return std::reduce(std::execution::par_unseq, sum.cbegin(), sum.cend());
	}

	void calculate_histogram_RSS_cpu(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		tbb::tick_count total1 = tbb::tick_count::now();

		std::vector<std::future<double>> workers(4);

		get_gauss_rss(res, histogramDensity, hist);

		// Run workers
		workers[0] = std::async(std::launch::async, &get_gauss_rss, std::ref(res), std::ref(histogramDensity), std::ref(hist));
		workers[1] = std::async(std::launch::async, &get_exp_rss, std::ref(res), std::ref(histogramDensity), std::ref(hist));
		workers[2] = std::async(std::launch::async, &get_poisson_rss, std::ref(res), std::ref(histogramDensity), std::ref(hist));
		workers[3] = std::async(std::launch::async, &get_uniform_rss, std::ref(res), std::ref(histogramDensity), std::ref(hist));

		// Agregate results results
		res.gauss_rss = workers[0].get();
		res.exp_rss = workers[1].get();
		res.poisson_rss = workers[2].get();
		res.uniform_rss = workers[3].get();

		tbb::tick_count total2 = tbb::tick_count::now();
		res.total_rss_time = (total2 - total1).seconds();
	}

	double get_gauss_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		ppr::rss::Distribution* dist = new ppr::rss::NormalDistribution(res.gauss_mean, res.gauss_stdev);
		// Compute RSS
		for (size_t i = 0; i < histogramDensity.size(); i++)
		{
			double d = (double)histogramDensity[i];
			dist->Push(d, (res.uniform_a + (i * hist.binSize)));
		}
		double result = dist->Get_RSS();

		// Free instance
		delete dist;

		return result;
	}

	double get_poisson_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		ppr::rss::Distribution* dist = new ppr::rss::PoissonDistribution(res.poisson_lambda);
		// Compute RSS
		for (size_t i = 0; i < histogramDensity.size(); i++)
		{
			double d = (double)histogramDensity[i];
			dist->Push(d, (res.uniform_a + (i * hist.binSize)));
		}
		double result = dist->Get_RSS();

		// Free instance
		delete dist;

		return result;
	}

	double get_exp_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		double mean = res.uniform_a;
		double stdev = res.gauss_mean - mean;

		ppr::rss::Distribution* dist = new ppr::rss::ExponentialDistribution(mean, stdev);
		// Compute RSS
		for (size_t i = 0; i < histogramDensity.size(); i++)
		{
			double d = (double)histogramDensity[i];
			dist->Push(d, (res.uniform_a + (i * hist.binSize)));
		}
		double result = dist->Get_RSS();

		// Free instance
		delete dist;

		return result;
	}

	double get_uniform_rss(SResult& res, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		double mean = res.uniform_a;
		double stdev = res.uniform_b - res.uniform_a;
		ppr::rss::Distribution* dist = new ppr::rss::UniformDistribution(mean, stdev);

		// Compute RSS
		for (size_t i = 0; i < histogramDensity.size(); i++)
		{
			double d = (double)histogramDensity[i];
			dist->Push(d, (res.uniform_a + (i * hist.binSize)));
		}
		double result = dist->Get_RSS();

		// Free instance
		delete dist;

		return result;
	}
}
