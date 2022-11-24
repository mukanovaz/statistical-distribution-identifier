#include "executor.h"
#include <algorithm>

namespace ppr::executor
{
	// https://www.cs.cmu.edu/afs/cs/academic/class/15499-s09/www/handouts/TBB-HPCC07.pdf
	class MinIndexBody {
		const std::vector<double> my_a;
	public:
		double value_of_min;
		size_t index_of_min;

		MinIndexBody(const std::vector<double> a) :
			my_a(a),
			value_of_min(FLT_MAX),
			index_of_min(-1)
		{}

		MinIndexBody(MinIndexBody& x, tbb::split) :
			my_a(x.my_a),
			value_of_min(FLT_MAX),
			index_of_min(-1)
		{}

		void operator()(const tbb::blocked_range<size_t>& r) {
			const std::vector<double> a = my_a;
			for (size_t i = r.begin(); i != r.end(); ++i) {
				double value = a[i];
				if (value < value_of_min) {
					value_of_min = value;
					index_of_min = i;
				}
			}
		}

		void join(const MinIndexBody& y) {
			if (y.value_of_min < value_of_min) {
				value_of_min = y.value_of_min;
				index_of_min = y.index_of_min;
			}
		}
	};

	double SumVectorOnCPU(tbb::task_arena& arena, std::vector<double> data)
	{
		double sum = 0;
		arena.execute([&]() {
			sum = tbb::parallel_reduce(tbb::blocked_range<std::vector<double>::iterator>(data.begin(), data.end()), 0.0,
				[](tbb::blocked_range<std::vector<double>::iterator> const& range, double init) {
					return std::accumulate(range.begin(), range.end(), init);
				}, std::plus<double>());
			});

		return sum;
	}

	SDataStat RunStatisticsOnGPU(SOpenCLConfig& opencl, SConfig& configuration, tbb::task_arena& arena, unsigned long data_count_for_gpu, unsigned long wg_count, double* data)
	{
		cl_int err = 0;

		cl::Buffer buf(opencl.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, data_count_for_gpu * sizeof(double), data, &err);
		cl::Buffer out_sum_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, wg_count * sizeof(double), nullptr, &err);
		cl::Buffer out_sumAbs_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, wg_count * sizeof(double), nullptr, &err);
		cl::Buffer out_min_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, wg_count * sizeof(double), nullptr, &err);
		cl::Buffer out_max_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, wg_count * sizeof(double), nullptr, &err);

		// Set method arguments
		err = opencl.kernel.setArg(0, buf);
		err = opencl.kernel.setArg(1, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(2, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(3, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(4, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(5, out_sum_buf);
		err = opencl.kernel.setArg(6, out_sumAbs_buf);
		err = opencl.kernel.setArg(7, out_min_buf);
		err = opencl.kernel.setArg(8, out_max_buf);

		// Result data
		std::vector<double> out_sum(wg_count);
		std::vector<double> out_sumAbs(wg_count);
		std::vector<double> out_min(wg_count);
		std::vector<double> out_max(wg_count);

		cl::CommandQueue cmd_queue(opencl.context, opencl.device, 0, &err);

		// Pass all data to GPU
		err = cmd_queue.enqueueNDRangeKernel(opencl.kernel, cl::NullRange, cl::NDRange(data_count_for_gpu), cl::NDRange(opencl.wg_size));
		err = cmd_queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, wg_count * sizeof(double), out_sum.data());
		err = cmd_queue.enqueueReadBuffer(out_sumAbs_buf, CL_TRUE, 0, wg_count * sizeof(double), out_sumAbs.data());
		err = cmd_queue.enqueueReadBuffer(out_min_buf, CL_TRUE, 0, wg_count * sizeof(double), out_min.data());
		err = cmd_queue.enqueueReadBuffer(out_max_buf, CL_TRUE, 0, wg_count * sizeof(double), out_max.data());

		cl::finish();

		// Agregate results on CPU
		double sum = ppr::executor::SumVectorOnCPU(arena, out_sum);
		double sumAbs = ppr::executor::SumVectorOnCPU(arena, out_sumAbs);
		MinIndexBody mib(out_min);
		ppr::executor::RunOnCPU<MinIndexBody>(arena, mib, 0, wg_count);

		return { 
			data_count_for_gpu,			// n
			sum,						// sum
			sumAbs,						// sumAbs
			0.0,						// max
			mib.value_of_min			// min
		};
	}


	
}