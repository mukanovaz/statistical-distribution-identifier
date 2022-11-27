#define _CRTDBG_MAP_ALLOC
#include<crtdbg.h>

#include "file_mapping.h"
#include "config.h"
#include "data.h"
#include "./smp/smp_solver.h"
#include "./sequential/seq_solver.h"
#include "./opencl/gpu_solver.h"

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

void GetDistString(SResult result)
{
	switch (result.dist) {
		case EDistribution::GAUSS: 
			std::cout << "Input data have 'Gauss/Normal distribution' with mean=" << result.gauss_mean << " and variance=" << result.gauss_variance << std::endl;
			break;
		case EDistribution::POISSON: 
			std::cout << "Input data have 'Poisson distribution' with lambda=" << result.poisson_lambda << result.gauss_variance << std::endl;
			break;
		case EDistribution::EXP: 
			std::cout << "Input data have 'Exponential distribution' with lambda=" << result.exp_lambda << result.gauss_variance << std::endl;
			break;
		case EDistribution::UNIFORM: 
			std::cout << "Input data have 'Uniform distribution' with a=" << result.uniform_a << " and b=" << result.uniform_b << std::endl;
			break;
		default: "Unknown";
	}
}

int main(int argc, char** argv) {
	ppr::SConfig conf;
	bool parse_result = parse_args(argc, argv, conf);
	if (!parse_result)
	{
		return ppr::EExitStatus::ARGS;
	}

	SResult result = run(conf);

	GetDistString(result);
	std::cout << "RSS [g/n:" << result.gauss_rss << " p:" << result.poisson_rss << " e:" << result.exp_rss << " u:" << result.uniform_rss << "]" << std::endl;

	_CrtDumpMemoryLeaks();

	getchar();
	return result.status;
}
