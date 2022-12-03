#include "watchdog.h"
#undef min
#undef max

#include <tbb/tick_count.h>

namespace ppr::watchdog
{
	void start_watchdog(SDataStat& stat, SHistogram& hist, int& stage,
		std::vector<int>& histogram, std::vector<double>& histogramDesity, int data_count)
	{
        std::thread watchdog([&]() {
			EExitStatus status = EExitStatus::SUCCESS;
			unsigned int n_last = 0;
			bool once = false;

			while (true)
			{
				switch (stage)
				{
				//	===== [Statistics] =====
				case 0:	
					if (n_last > stat.n)						// Number of observing data is not increaing
					{
						status = EExitStatus::WD_STAT_WRONG_N;
					}
					n_last = stat.n;
					if (stat.min > stat.max)					// Min is bigger that max
					{
						status = EExitStatus::WD_STAT_MIN_MAX;
					}
					break;
				//	===== [Frequency histogram] =====
				case 1:	
					if (hist.binCount == 0)						// Bin count is zero
					{
						status = EExitStatus::WD_HIST_BIN_COUNT;
					}
					if (histogram.size() == 0)					// Histogram vector has no elements
					{
						status = EExitStatus::WD_HIST_SIZE;
					}
					break;
				//	===== [Density histogram] =====
				case 2:	
					if (std::all_of(histogram.begin(), histogram.end(), [](int i) { return i == 0; }))				// Histogram is all of zeros after computing histogram
					{
						status = EExitStatus::WD_HIST_ALL_ZERO;
					}
					break;
				//	===== [RSS] =====
				case 3:		
					if (std::all_of(histogramDesity.begin(), histogramDesity.end(), [](int i) { return i == 0; }))	// Histogram Density is all of zeros after computing histogram
					{
						status = EExitStatus::WD_DHIST_ALL_ZERO;
					}
					break;
				//	===== [END] =====
				case 4:
					return;
				default:
					status = EExitStatus::UNKNOWN;
					return;
				}

				if (status != EExitStatus::SUCCESS)
				{
					std::cerr << "> [WARNING] Watchdog found some problem in computing process - exit status[" << status << "]" << std::endl;
					std::exit(status);
				}

				std::this_thread::sleep_for(std::chrono::seconds(WATCHDOG_INTERVAL_SEC));
			}

        });

		watchdog.detach();
	}
}