#pragma once
#include <thread>
#include "data.h"
#include "config.h"

namespace ppr::watchdog
{
	/// <summary>
	/// Function is creatinf one thread, which is checking program status in infinite loop
	/// </summary>
	/// <param name="stat">Statistics structure</param>
	/// <param name="hist">Histogram configuration structure</param>
	/// <param name="stage">Current program stage</param>
	/// <param name="histogram">Histogram vector reference</param>
	/// <param name="histogramDesity">Density histogram vector reference</param>
	/// <param name="data_count">Data count in a file</param>
	std::thread start_watchdog(SConfig& config, SDataStat& stat, SHistogram& hist, int& stage,
		std::vector<int>& histogram, std::vector<double>& histogramDesity, long data_count);
}