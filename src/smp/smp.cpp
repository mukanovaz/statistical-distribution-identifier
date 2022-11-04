#include "smp.h"

namespace ppr::smp
{
	SResult run(SConfig& configuration)
	{
		SResult result;
		result.dist = EDistribution::GAUSS;
		return result;
	}
}