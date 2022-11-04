#include "file_mapping.h"
#include "config.h"
#include "data.h"
#include "./smp/smp.h"
#include "./all/all.h"

namespace ppr
{
	SResult run(SConfig& configuration)
	{
		/*switch (configuration.mode) {
		case ERun_mode::SMP:  
			return smp::run(configuration);
		case ERun_mode::ALL:  
			return all::run(configuration);

		default:
			return SResult::error_res(EExitStatus::UNKNOWN);
		}*/
		return SResult::error_res(EExitStatus::UNKNOWN);
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

	/*std::unique_ptr<HANDLE> file = create_file(argv[1]);
	std::unique_ptr<HANDLE> mapping = map_file(*file);
	const double* data = get_data(*file, *mapping);
	unmap_file(data, *file, *mapping);

	mapping.reset();
	file.reset();*/

	getchar();
	return result.status;
}
