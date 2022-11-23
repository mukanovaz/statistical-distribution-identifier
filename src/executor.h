#pragma once
#undef min
#undef max

#include <tbb/tick_count.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/blocked_range.h>
#include "rss/statistics.cpp"
#include "histogram/histogram.cpp"

namespace ppr::executor
{
	template <typename T>
	double RunOnCPU(tbb::task_arena& arena, T& class_to_execute, int begin, int end)
	{
		tbb::tick_count t0 = tbb::tick_count::now();
		arena.execute([&]() {
			tbb::parallel_reduce(tbb::blocked_range<std::size_t>(begin, end), class_to_execute);
			});
		tbb::tick_count t1 = tbb::tick_count::now();

		return (t1 - t0).seconds();
	}
}
