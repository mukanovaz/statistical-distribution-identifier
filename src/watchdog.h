#pragma once

#include <thread>
#include "data.h"
#include "config.h"

namespace ppr::watchdog
{
	void start_watchdog(SDataStat& stat, SHistogram& hist, int& stage,
		std::vector<int>& histogram, std::vector<double>& histogramDesity, int data_count);
}