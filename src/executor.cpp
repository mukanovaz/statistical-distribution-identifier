#include "executor.h"
#include <algorithm>

namespace ppr::executor
{
	// https://www.cs.cmu.edu/afs/cs/academic/class/15499-s09/www/handouts/TBB-HPCC07.pdf
	class MinParallel {
		const std::vector<double> my_a;
	public:
		double value_of_min;

		MinParallel(const std::vector<double> a) :
			my_a(a),
			value_of_min(FLT_MAX)
		{}

		MinParallel(MinParallel& x, tbb::split) :
			my_a(x.my_a),
			value_of_min(FLT_MAX)
		{}

		void operator()(const tbb::blocked_range<size_t>& r) {
			const std::vector<double> a = my_a;
			for (size_t i = r.begin(); i != r.end(); ++i) {
				double value = a[i];
				value_of_min = std::min({ value_of_min , value });
			}
		}

		void join(const MinParallel& y) {
			value_of_min = std::min({ value_of_min , y.value_of_min });
		}
	};

	class MaxParallel {
		const std::vector<double> my_a;
	public:
		double value_of_max;

		MaxParallel(const std::vector<double> a) :
			my_a(a),
			value_of_max(FLT_MIN)
		{}

		MaxParallel(MaxParallel& x, tbb::split) :
			my_a(x.my_a),
			value_of_max(FLT_MIN)
		{}

		void operator()(const tbb::blocked_range<size_t>& r) {
			const std::vector<double> a = my_a;
			for (size_t i = r.begin(); i != r.end(); ++i) {
				double value = a[i];
				value_of_max = std::max({ value_of_max , value });
			}
		}

		void join(const MaxParallel& y) {
			value_of_max = std::max({ value_of_max , y.value_of_max });
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

	SDataStat RunStatisticsOnGPU(SOpenCLConfig& opencl, SConfig& configuration, tbb::task_arena& arena, double* data)
	{
		cl_int err = 0;

		cl::Buffer buf(opencl.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, opencl.data_count_for_gpu * sizeof(double), data, &err);
		cl::Buffer out_sum_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, opencl.wg_count * sizeof(double), nullptr, &err);
		cl::Buffer out_negative_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, opencl.wg_count * sizeof(int), nullptr, &err);
		cl::Buffer out_min_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, opencl.wg_count * sizeof(double), nullptr, &err);
		cl::Buffer out_max_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, opencl.wg_count * sizeof(double), nullptr, &err);

		// Set method arguments
		err = opencl.kernel.setArg(0, buf);
		err = opencl.kernel.setArg(1, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(2, opencl.wg_size * sizeof(int), nullptr);
		err = opencl.kernel.setArg(3, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(4, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(5, out_sum_buf);
		err = opencl.kernel.setArg(6, out_negative_buf);
		err = opencl.kernel.setArg(7, out_min_buf);
		err = opencl.kernel.setArg(8, out_max_buf);
		
		// Result data
		std::vector<double> out_sum(opencl.wg_count);
		std::vector<int> out_negative(opencl.wg_count);
		std::vector<double> out_min(opencl.wg_count);
		std::vector<double> out_max(opencl.wg_count);

		cl::CommandQueue cmd_queue(opencl.context, opencl.device, 0, &err);

		// Pass all data to GPU
		err = cmd_queue.enqueueNDRangeKernel(opencl.kernel, cl::NullRange, cl::NDRange(opencl.data_count_for_gpu), cl::NDRange(opencl.wg_size));
		err = cmd_queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, opencl.wg_count * sizeof(double), out_sum.data());
		err = cmd_queue.enqueueReadBuffer(out_negative_buf, CL_TRUE, 0, opencl.wg_count * sizeof(int), out_negative.data());
		err = cmd_queue.enqueueReadBuffer(out_min_buf, CL_TRUE, 0, opencl.wg_count * sizeof(double), out_min.data());
		err = cmd_queue.enqueueReadBuffer(out_max_buf, CL_TRUE, 0, opencl.wg_count * sizeof(double), out_max.data());

		cl::finish();
		
		// Agregate results on CPU
		bool isNegative = std::find(std::begin(out_negative), std::end(out_negative), 1) != std::end(out_negative);
		double sum = ppr::executor::SumVectorOnCPU(arena, out_sum);
		//double negative = ppr::executor::SumVectorOnCPU(arena, out_negative);
		MinParallel minp(out_min);
		ppr::executor::RunOnCPU<MinParallel>(arena, minp, 0, opencl.wg_count);
		MaxParallel maxp(out_max);
		ppr::executor::RunOnCPU<MaxParallel>(arena, maxp, 0, opencl.wg_count);

		return { 
			opencl.data_count_for_gpu,	// n
			sum,						// sum
			maxp.value_of_max,			// max
			minp.value_of_min,			// min
			0.0,						// mean
			0.0,						// variance
			isNegative
		};
	}

	void RunHistogramOnGPU(SOpenCLConfig& opencl, SDataStat& data_stat, SHistogram& histogram, tbb::task_arena& arena, double* data, std::vector<int>& freq_buckets)
	{
		cl_int err = 0;

		std::vector<cl_uint> out_histogram(2 * histogram.binCount, 0);

		cl::Buffer buf(opencl.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, opencl.data_count_for_gpu * sizeof(double), data, &err);
		cl::Buffer out_sum_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, out_histogram.size() * sizeof(cl_uint), out_histogram.data(), &err);
		cl::Buffer out_var_buf(opencl.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, opencl.wg_count * sizeof(double), nullptr, &err);

		// Set method arguments
 		err = opencl.kernel.setArg(0, buf);
		err = opencl.kernel.setArg(1, opencl.wg_size * sizeof(double), nullptr);
		err = opencl.kernel.setArg(2, out_sum_buf);
		err = opencl.kernel.setArg(3, out_var_buf);
		err = opencl.kernel.setArg(4, sizeof(double), &data_stat.mean);
		err = opencl.kernel.setArg(5, sizeof(double), &data_stat.min);
		err = opencl.kernel.setArg(6, sizeof(double), &histogram.scaleFactor);
		err = opencl.kernel.setArg(7, sizeof(double), &histogram.binSize);
		err = opencl.kernel.setArg(8, sizeof(double), &histogram.binCount);

		// Result data
		std::vector<double> out_var(opencl.wg_count);

		cl::CommandQueue cmd_queue(opencl.context, opencl.device, 0, &err);

		// Pass all data to GPU
		err = cmd_queue.enqueueNDRangeKernel(opencl.kernel, cl::NullRange, cl::NDRange(opencl.data_count_for_gpu), cl::NDRange(opencl.wg_size));
		err = cmd_queue.enqueueReadBuffer(out_sum_buf, CL_TRUE, 0, out_histogram.size() * sizeof(cl_uint), out_histogram.data());
		err = cmd_queue.enqueueReadBuffer(out_var_buf, CL_TRUE, 0, opencl.wg_count * sizeof(double), out_var.data());

		cl::finish();

		for (int i = 0; i < histogram.binCount; i++)
		{
			const size_t value = static_cast<size_t>(out_histogram.at(2 * i)) + static_cast<size_t>(out_histogram.at(2 * i + 1)) * sizeof(cl_uint);
			freq_buckets[i] += value;
		}

		// Agregate results on CPU
		double var = ppr::executor::SumVectorOnCPU(arena, out_var);
		data_stat.variance += var;
	}

	void CalculateHistogramRSSOnCPU(SResult& res, tbb::task_arena& arena, std::vector<double>& histogramDensity, SHistogram& hist)
	{
		tbb::tick_count total1 = tbb::tick_count::now();

		ppr::rss::Distribution* gauss = new ppr::rss::NormalDistribution(res.gauss_mean, res.gauss_stdev, res.gauss_variance);
		ppr::rss::Distribution* poisson = new ppr::rss::PoissonDistribution(res.poisson_lambda);
		ppr::rss::Distribution* exp = new ppr::rss::ExponentialDistribution(res.exp_lambda);
		ppr::rss::Distribution* uniform = new ppr::rss::UniformDistribution(res.uniform_a, res.uniform_b);

		ppr::rss::RSSParallel gauss_rss(gauss, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel poisson_rss(poisson, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel exp_rss(exp, histogramDensity, hist.binSize);
		ppr::rss::RSSParallel uniform_rss(uniform, histogramDensity, hist.binSize);

		double t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, gauss_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Gauss RSS:\t" << t << "\tsec." << std::endl;

		t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, poisson_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Poisson RSS:\t" << t << "\tsec." << std::endl;

		t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, exp_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Expon RSS:\t" << t << "\tsec." << std::endl;

		t = ppr::executor::RunOnCPU<ppr::rss::RSSParallel>(arena, uniform_rss, 0, static_cast<int>(hist.binCount));
		std::cout << "Uniform RSS:\t" << t << "\tsec." << std::endl;

		res.gauss_rss = gauss->Get_RSS();
		res.poisson_rss = poisson->Get_RSS();
		res.exp_rss = exp->Get_RSS();
		res.uniform_rss = uniform->Get_RSS();

		// Free allocations
		delete gauss;
		delete poisson;
		delete exp;
		delete uniform;

		tbb::tick_count total2 = tbb::tick_count::now();

		std::cout << "Total RSS:\t" << (total2 - total1).seconds() << "\tsec." << std::endl;
	}

	void AnalyzeResults(SResult& res)
	{
		// Find min RSS value
		std::array<double, 4> rss = { res.gauss_rss, res.poisson_rss, res.exp_rss, res.uniform_rss };
		std::sort(rss.begin(), rss.end());

		bool canBePoisson = !res.isNegative && res.isInteger;
		bool canBeExp = !res.isNegative && !res.isInteger;

		if (res.uniform_rss == rss[0])
		{
			res.dist = EDistribution::UNIFORM;
		}
		else if ((res.poisson_rss == rss[0] && canBePoisson) || (res.poisson_rss == rss[1] && canBePoisson))
		{
			res.dist = EDistribution::POISSON;
		}
		else if ((res.exp_rss == rss[0] && canBeExp) || (res.exp_rss == rss[1] && canBeExp))
		{
			res.dist = EDistribution::EXP;
		}
		else if (res.gauss_rss == rss[0])
		{
			res.dist = EDistribution::GAUSS;
		}
		res.status = EExitStatus::SUCCESS;
	}

	void ComputePropabilityDensityOfHistogram(SHistogram& hist, std::vector<int>& bucket_frequency, std::vector<double>& bucket_density, double count)
	{
		for (unsigned int i = 0; i < hist.binCount; i++)
		{
			double next_edge = hist.min + (hist.binSize * (static_cast<double>(i) + 1.0));
			double curr_edge = hist.min + (hist.binSize * static_cast<double>(i));
			double diff = next_edge - curr_edge;
			bucket_density[i] = bucket_frequency[i] / diff / count;
		}
	}
	
}