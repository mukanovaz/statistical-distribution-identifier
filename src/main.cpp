#include "include/file_mapping.h"
#include "include/config.h"
#include "include/data.h"
#include "include/smp_solver.h"
#include "include/seq_solver.h"
#include "include/seq_solver.h"
#include "include/solver.h"

#include <iostream>

namespace ppr
{
	SResult run(SConfig& configuration)
	{
		switch (configuration.mode) {
		case ERun_mode::SEQ:
			return seq::run(configuration);
		case ERun_mode::SMP:  
			return solver::run(configuration);
		//case ERun_mode::ALL:  
		//	return gpu::run(configuration);
		//case ERun_mode::CL:
		//	return gpu::run(configuration);
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
			std::cout << "> Input data have 'Poisson distribution' with lambda=" << result.poisson_lambda << std::endl;
			break;
		case EDistribution::EXP: 
			std::cout << "> Input data have 'Exponential distribution' with lambda=" << result.exp_lambda << std::endl;
			break;
		case EDistribution::UNIFORM: 
			std::cout << "> Input data have 'Uniform distribution' with a=" << result.uniform_a << " and b=" << result.uniform_b << std::endl;
			break;
		default: "Unknown";
			break;
	}
}

int main(int argc, char* argv[])
{
	ppr::SConfig conf;
	bool parse_result = parse_args(argc, argv, conf);
	if (!parse_result)
	{
		return ppr::EExitStatus::ARGS;
	}

	std::string opt = conf.use_optimalization ? "TRUE" : "FALSE";

	std::cout << "\t\t\t[Initial parameters]" << std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;
	std::cout << "> File:\t\t\t\t" << argv[1] << std::endl;
	std::cout << "> Mode:\t\t\t\t" << ppr::print_mode(conf.mode) << std::endl;
	if (conf.cl_devices_name.size() != 0 && conf.mode == ppr::ERun_mode::CL)
	{
		std::cout << "> Devices:\t\t\t" << std::endl;

		for (size_t i = 0; i < conf.cl_devices_name.size(); i++)
		{
			std::cout << "> \t\t\t\t" << conf.cl_devices_name[i] << std::endl;
		}
	}
	std::cout << "> Number of threads:\t\t" << conf.thread_count << std::endl;
	std::cout << "> Optimalization:\t\t" << opt << std::endl;
	std::cout << "> Watchdog timer:\t\t" << conf.watchdog_interval << " sec" << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "> Started .." << std::endl;
	std::cout << std::endl;
	SResult result = run(conf);
	std::cout << std::endl;

	std::cout << "\t\t\t[Results]" << std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;
	std::cout << "> Gauss RSS:\t\t\t" << result.gauss_rss << std::endl;
	std::cout << "> Poisson RSS:\t\t\t" << result.poisson_rss << std::endl;
	std::cout << "> Exponential RSS:\t\t" << result.exp_rss << std::endl;
	std::cout << "> Uniform RSS:\t\t\t" << result.uniform_rss << std::endl;

	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "\t\t\t[Time]" << std::endl;
	std::cout << "---------------------------------------------------------------------" << std::endl;
	std::cout << "> Statistics computing time:\t" << result.total_stat_time << " sec." << std::endl;
	std::cout << "> Histogram computing time:\t" << result.total_hist_time << " sec." << std::endl;
	std::cout << "> RSS computing time:\t\t" << result.total_rss_time << " sec." << std::endl;
	std::cout << "> TOTAL TIME:\t\t\t" << result.total_time << " sec." << std::endl;
	std::cout << std::endl;
	get_dist_string(result);

	//getchar();
	return result.status;
}
