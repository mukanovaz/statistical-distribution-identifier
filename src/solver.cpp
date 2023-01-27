#include "include/solver.h"
#include "include/file_mapper.h"
#include "include/watchdog.h"

#include <execution>

namespace ppr::solver
{
	DWORDLONG getAvailPhysMem()
	{
		MEMORYSTATUSEX status;
		status.dwLength = sizeof(status);
		GlobalMemoryStatusEx(&status);
		return status.ullAvailPhys;
	}

	SResult run(SConfig& configuration)
	{		
		//  ================ [Allocations]
		tbb::tick_count total1;
		total1 = tbb::tick_count::now();
		tbb::tick_count total2;
		tbb::tick_count t0;
		tbb::tick_count t1;
		int stage = 0;
		SDataStat stat;
		SResult res;
		SHistogram hist;
		std::vector<int> histogramFreq(0);			// Will resize after collecting statistics
		std::vector<double> histogramDensity(0);	// Will resize after collecting statistics
		DWORD64 data_count = 0;

		DWORDLONG ram_mem = getAvailPhysMem() > MAX_FILE_SIZE_MEM_2gb ? getAvailPhysMem() - 1000000000 : getAvailPhysMem();

		std::vector<cl::Device> devices;
		ppr::gpu::find_opencl_devices(devices, configuration.cl_devices_name);

		//  ================ [Map input file]
		File_mapper* mapper = File_mapper::get_instance();
		mapper->init(configuration.input_fn);

		const unsigned long long file_len = mapper->get_file_len();

		if (configuration.mode == ERun_mode::SMP)
		{
			DWORD64 ram_per_thread = MAX_FILE_SIZE_MEM_600mb / configuration.thread_count;
			data_count = ram_per_thread - (ram_per_thread % mapper->get_granularity());
		}
		else
		{
			DWORD64 mem_per_thread = MAX_FILE_SIZE_MEM_300mb / devices.size();
			data_count = mem_per_thread - (mem_per_thread % mapper->get_granularity());
		}

		//  ================ [Start Watchdog]
		std::thread watchdog = ppr::watchdog::start_watchdog(configuration, stat, hist, stage, histogramFreq, histogramDensity, data_count);

		//  ================ [Get statistics]
		t0 = tbb::tick_count::now();
		compute_statistics(devices, configuration, stat, file_len, data_count);
		t1 = tbb::tick_count::now();

		res.total_stat_time = (t1 - t0).seconds();

		//  ================ [Fit params using Maximum likelihood estimation]

		res.isNegative = stat.min < 0;
		res.isInteger = std::floor(stat.sum) == stat.sum;

		// Find mean
		stat.mean = stat.sum / stat.n;

		// Poisson likelihood estimators
		res.poisson_lambda = stat.sum / stat.n;

		//  ================ [Create frequency histogram]

		if (configuration.mode != ERun_mode::SMP)
		{
			DWORD64 mem_per_thread = MAX_FILE_SIZE_MEM_600mb / devices.size();
			data_count = mem_per_thread - (mem_per_thread % mapper->get_granularity());
		}

		// Find histogram limits
		double bin_count = 0.0;
		double bin_size = 0.0;

		// If data can belongs to poisson distribution, we should use integer intervals
		if (!res.isNegative && res.isInteger && res.poisson_lambda > 0)
		{
			hist.binCount = static_cast<int>(stat.max - stat.min);
			hist.binSize = 1.0;
		}
		else
		{
			hist.binCount = static_cast<int>(log2(stat.n)) + 2;
			hist.binSize = (stat.max - stat.min) / (hist.binCount - 1);
		}
		hist.scaleFactor = (hist.binCount) / (stat.max - stat.min);

		// Allocate memmory
		histogramFreq.resize(static_cast<int>(hist.binCount));
		histogramDensity.resize(static_cast<int>(hist.binCount));

		t0 = tbb::tick_count::now();
		compute_histogram(devices, hist, configuration, stat, file_len, data_count, histogramFreq);
		t1 = tbb::tick_count::now();

		res.total_hist_time = (t1 - t0).seconds();

		// Close file
		mapper->close_all();

		//  ================ [Fit params using Maximum likelihood estimation]

		// Find variance
		stat.variance = stat.variance / stat.n;

		// Gauss maximum likelihood estimators
		res.gauss_mean = stat.mean;
		res.gauss_variance = stat.variance;
		res.gauss_stdev = sqrt(stat.variance);

		// Exponential maximum likelihood estimators
		res.exp_lambda = stat.n / stat.sum;

		// Uniform likelihood estimators
		res.uniform_a = stat.min;
		res.uniform_b = stat.max;

		//  ================ [Create density histogram]
		stage = 2;
		ppr::executor::compute_propability_density_histogram(hist, histogramFreq, histogramDensity, stat.n);

		//	================ [Calculate RSS]
		stage = 3;
		ppr::parallel::calculate_histogram_RSS_cpu(res, histogramDensity, hist);

		//	================ [Analyze Results]
		ppr::executor::analyze_results(res);

		total2 = tbb::tick_count::now();

		res.total_time = (total2 - total1).seconds();
		stage = 4;

		print_stat(stat, res);

		// Wait until watchdog will finish
		watchdog.join();
		return res;
	}

	void compute_histogram(std::vector<cl::Device> devices, SHistogram& hist, SConfig& configuration, SDataStat& stat, const unsigned long long file_len, DWORD64 data_count,
		std::vector<int>& histogram)
	{
		int index = 0;
		int workers_count = configuration.mode == ERun_mode::SMP ? configuration.thread_count : devices.size();

		std::vector<std::future<std::tuple<std::vector<int>, double>>> workers_hist(workers_count);

		for (unsigned long long offset = 0; offset < file_len; offset += data_count)
		{
			ppr::gpu::SOpenCLConfig opencl;
			DWORD high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
			DWORD low = static_cast<DWORD>(offset & 0xFFFFFFFFul);

			if (offset + data_count > file_len) {
				data_count = static_cast<int>(file_len - offset);
			}

			opencl.high = high;
			opencl.low = low;
			opencl.data_count = data_count;

			if (configuration.mode != ERun_mode::SMP)
			{
				bool res = ppr::gpu::init_opencl(devices[index], HIST_KERNEL, HIST_KERNEL_NAME, opencl);

				if (!res)
				{
					return;
					// TODO!!!
				}
				ppr::parallel::Hist_processing_unit unit(hist, configuration, opencl, stat);
				workers_hist[index] = std::async(std::launch::async, &ppr::parallel::Hist_processing_unit::run_on_GPU, unit);
			}
			else
			{
				ppr::parallel::Hist_processing_unit unit(hist, configuration, opencl, stat);
				workers_hist[index] = std::async(std::launch::async, &ppr::parallel::Hist_processing_unit::run_on_CPU, unit);
			}

			if (index == workers_count - 1)
			{
				for (auto& worker : workers_hist)
				{
					auto [vector, variance] = worker.get();
					stat.variance += variance;
					std::transform(std::execution::par, histogram.begin(), histogram.end(), vector.begin(), histogram.begin(), std::plus<int>());
				}

				index = 0;
				continue;
			}
			index++;
		}

		for (auto& worker : workers_hist)
		{
			if (!worker.valid())
			{
				continue;
			}

			auto [vector, variance] = worker.get();
			stat.variance += variance;
			std::transform(histogram.begin(), histogram.end(), vector.begin(), histogram.begin(), std::plus<int>());
		}
	}

	void compute_statistics(std::vector<cl::Device> devices, SConfig& configuration, SDataStat& stat, const unsigned long long file_len, DWORD64 data_count)
	{
		int index_stat = 0;
		int workers_count = configuration.mode == ERun_mode::SMP ? configuration.thread_count : devices.size();

		std::vector<std::future<SDataStat>> workers_stat(workers_count);

		for (unsigned long long offset = 0; offset < file_len; offset += data_count)
		{
			ppr::gpu::SOpenCLConfig opencl;
			DWORD high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
			DWORD low = static_cast<DWORD>(offset & 0xFFFFFFFFul);

			if (offset + data_count > file_len) {
				data_count = static_cast<int>(file_len - offset);
			}

			opencl.high = high;
			opencl.low = low;
			opencl.data_count = data_count;

			ppr::parallel::Stat_processing_unit unit(configuration, opencl);

			if (configuration.mode != ERun_mode::SMP)
			{
				bool res = ppr::gpu::init_opencl(devices[index_stat], STAT_KERNEL, STAT_KERNEL_NAME, opencl);

				if (!res)
				{
					return;
					// TODO!!!
				}
				ppr::parallel::Stat_processing_unit unit(configuration, opencl);
				workers_stat[index_stat] = std::async(std::launch::async, &ppr::parallel::Stat_processing_unit::run_on_GPU, unit);
			}
			else
			{
				ppr::parallel::Stat_processing_unit unit(configuration, opencl);
				workers_stat[index_stat] = std::async(std::launch::async, &ppr::parallel::Stat_processing_unit::run_on_CPU, unit);
			}
			
			if (index_stat == workers_count - 1)
			{
				for (auto& worker : workers_stat)
				{
					SDataStat local_stat = worker.get();
					stat.sum += local_stat.sum;
					stat.n += local_stat.n;
					stat.max = std::max({ stat.max, local_stat.max });
					stat.min = std::min({ stat.min, local_stat.min });
				}

				index_stat = 0;
				continue;
			}
			index_stat++;
		}

		for (auto& worker : workers_stat)
		{
			if (!worker.valid())
			{
				continue;
			}
			SDataStat local_stat = worker.get();
			stat.sum += local_stat.sum;
			stat.n += local_stat.n;
			stat.max = std::max({ stat.max, local_stat.max });
			stat.min = std::min({ stat.min, local_stat.min });
		}
	}


}
