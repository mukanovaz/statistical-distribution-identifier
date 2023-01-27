#pragma once
#include "include/data.h"
#include "include/config.h"
#include "include/executor.h"

namespace ppr::seq
{	
	/// <summary>
	/// Starting function which runs sequentially
	/// </summary>
	/// <param name="configuration">Program configuration structure</param>
	/// <returns>Computing results</returns>
	SResult run(SConfig& configuration);
}
