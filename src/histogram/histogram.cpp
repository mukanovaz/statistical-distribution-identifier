#include <cmath>
#include <vector>

namespace ppr::hist
{
	class Histogram
	{
		private:
			double BinSize;
			double Min;
			std::vector<uint64_t> Buckets;

		public:
			Histogram(std::vector<uint64_t>& buckets, double bin_size, double min)
				: Buckets(buckets), BinSize(bin_size), Min(min)
			{
				/*double bin_count = log2(mapping.get_count()) + 1;
				double bin_size = (stat.Get_Max() - stat.Get_Min()) / bin_count;
				std::vector<uint64_t> buckets(static_cast<int>(bin_count) + 1);*/
			}

            void Push(double x)
            {
				double position = ((x - Min) / BinSize);
				Buckets[static_cast<int>(position)]++;
            }

			const std::vector<uint64_t> Get_buckets() const
			{
				return Buckets;
			}
	};
}