#pragma once
#include "../data.h"
#include "../config.h"
#include "../file_mapping.h"
#include "../executor.h"

namespace ppr::seq
{	
	/// <summary>
	/// Starting function which runs sequentially
	/// </summary>
	/// <param name="configuration">Program configuration structure</param>
	/// <returns>Computing results</returns>
	SResult run(SConfig& configuration);
}
