#include "include/file_mapping.h"
#include "include/config.h"
#include "include/data.h"
#include "include/smp_solver.h"
#include "include/seq_solver.h"
#include "include/gpu_solver.h"

#include <iostream>

namespace ppr
{
	SResult run(SConfig& configuration)
	{
		switch (configuration.mode) {
		case ERun_mode::SMP:  
			return parallel::run(configuration);
		case ERun_mode::ALL:  
			return gpu::run(configuration);
		default:
			return SResult::error_res(EExitStatus::UNKNOWN);
		}
	}
}

void get_dist_string(SResult result)
{
	switch (result.dist) {
		case EDistribution::GAUSS: 
			std::cout << "> Input data have 'Gauss/Normal distribution' with mean=" << result.gauss_mean << " and variance=" << result.gauss_variance << std::endl;
			break;
		case EDistribution::POISSON: 
			std::cout << "> Input data have 'Poisson distribution' with lambda=" << result.poisson_lambda << result.gauss_variance << std::endl;
			break;
		case EDistribution::EXP: 
			std::cout << "> Input data have 'Exponential distribution' with lambda=" << result.exp_lambda << result.gauss_variance << std::endl;
			break;
		case EDistribution::UNIFORM: 
			std::cout << "> Input data have 'Uniform distribution' with a=" << result.uniform_a << " and b=" << result.uniform_b << std::endl;
			break;
		default: "Unknown";
			break;
	}
}

int main(int argc, char** argv) {
	ppr::SConfig conf;
	bool parse_result = parse_args(argc, argv, conf);
	if (!parse_result)
	{
		return ppr::EExitStatus::ARGS;
	}

	SYSTEM_INFO sysInfo;

	//GetSystemInfo(&sysInfo);
	//printf("%s %d\n\n", "PageSize[Bytes] :", sysInfo.dwPageSize);

	std::string mode = conf.mode == ppr::ERun_mode::SMP ? "smp" : "all";
	std::string opt = conf.use_optimalization ? "TRUE" : "FALSE";

	std::cout << "\t\t\t[Initial parameters]" << std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;
	std::cout << "> Mode:\t\t\t\t" << mode << std::endl;
	std::cout << "> Number of threads:\t\t" << conf.thread_count << std::endl;
	std::cout << "> Optimalization:\t\t" << opt << std::endl;
	std::cout << "> Watchdog timer:\t\t" << conf.watchdog_interval << " sec" << std::endl;
	std::cout << "> Thread per core:\t\t" << conf.thread_per_core << " threads" << std::endl;
	std::cout << "> Statistics timeout:\t\t" << conf.stat_timeout << " sec" << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "> Starting .." << std::endl;
	SResult result = run(conf);
	std::cout << "> Finish" << std::endl;
	std::cout << std::endl;

	std::cout << "\t\t\t[Time]" << std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;
	std::cout << "> Statistics computing time:\t" << result.total_stat_time << " sec." << std::endl;
	std::cout << "> Histogram computing time:\t" << result.total_hist_time << " sec." << std::endl;
	std::cout << "> RSS computing time:\t\t" << result.total_rss_time << " sec." << std::endl;
	std::cout << "> TOTAL TIME:\t\t\t" << result.total_time << " sec." << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "\t\t\t[Results]" << std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;
	std::cout << "> Gauss RSS:\t\t\t" << result.gauss_rss << std::endl;
	std::cout << "> Poisson RSS:\t\t\t" << result.poisson_rss << std::endl;
	std::cout << "> Exponential RSS:\t\t" << result.exp_rss << std::endl;
	std::cout << "> Uniform RSS:\t\t\t" << result.uniform_rss << std::endl;
	std::cout << std::endl;
	std::cout << "> Contains negative:\t\t" << result.isNegative << std::endl;
	std::cout << "> All integers:\t\t\t" << result.isInteger << std::endl;
	std::cout << std::endl;
	get_dist_string(result);

	delete conf.input_fn;

	getchar();
	return result.status;
}
