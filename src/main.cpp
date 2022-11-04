#include "file_mapping.h"
#include "config.h"
#include "data.h"
#include "./smp/smp.h"
#include "./all/all.h"
#include "./sequential/seq_solver.h"

namespace ppr
{
	SResult run(SConfig& configuration)
	{
		switch (configuration.mode) {
		case ERun_mode::SMP:  
			return seq::run(configuration);
		///*case ERun_mode::ALL:  
		//	return all::run(configuration);*/

		default:
			return SResult::error_res(EExitStatus::UNKNOWN);
		}
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


	getchar();
	return result.status;
}
