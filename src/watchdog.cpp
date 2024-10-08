#include "include/watchdog.h"
#undef min
#undef max

#include <tbb/tick_count.h>

namespace ppr::watchdog
{
	bool is_all_zeros(double n) {
		return (n == 0);
	}

	std::thread start_watchdog(SConfig& config, SDataStat& stat, SHistogram& hist, int& stage,
		std::vector<int>& histogram, std::vector<double>& histogramDesity, long data_count)
	{
        std::thread watchdog([&]() {
			EExitStatus status = EExitStatus::SUCCESS;
			unsigned long long n_last = 0;

			while (true)
			{
				switch (stage)
				{
				//	===== [Statistics] =====
				case 0:	
					if (n_last > stat.n)						// Number of observing data is not increaing
					{
						ppr::print_error("Wrong number of processing data.");
						status = EExitStatus::WD_STAT_WRONG_N;
					}
					n_last = stat.n;
					if (stat.min > stat.max)					// Min is bigger that max
					{
						ppr::print_error("Minumum value cannot be bigger, that maximum.");
						status = EExitStatus::WD_STAT_MIN_MAX;
					}
					break;
				//	===== [Frequency histogram] =====
				case 1:	
					if (stat.n > data_count)						// Number of observing data is wrong
					{
						ppr::print_error("Wrong number of processing data.");
						status = EExitStatus::WD_STAT_WRONG_N;
					}
					if (hist.binCount == 0)						// Bin count is zero
					{
						ppr::print_error("Histogram 'bin count' is zero.");
						status = EExitStatus::WD_HIST_BIN_COUNT;
					}
					if (histogram.size() == 0)					// Histogram vector has no elements
					{
						ppr::print_error("Frequency histogram not exists.");
						status = EExitStatus::WD_HIST_SIZE;
					}
					break;
				//	===== [Density histogram] =====
				case 2:	
					if (std::all_of(histogram.begin(), histogram.end(), is_all_zeros))
					{
						ppr::print_error("Frequency Histogram vector contains only zeros after computing histogram.");
						status = EExitStatus::WD_HIST_ALL_ZERO;
					}
					break;
				//	===== [RSS] =====
				case 3:		
					if (std::all_of(histogramDesity.begin(), histogramDesity.end(), is_all_zeros))
					{
						ppr::print_error("Density Histogram vector contains only zeros after computing histogram.");
						status = EExitStatus::WD_DHIST_ALL_ZERO;
					}
					break;
				//	===== [END] =====
				case 4:
					return;
				default:
					ppr::print_error("Unknown error.");
					status = EExitStatus::UNKNOWN;
					return;
				}

				if (status != EExitStatus::SUCCESS)
				{
					ppr::print_error("Watchdog found some problem in computing process.");
					std::exit(status);
				}

				std::this_thread::sleep_for(std::chrono::seconds(config.watchdog_interval));
			}

        });

		return watchdog;
		//watchdog.detach();
	}
}