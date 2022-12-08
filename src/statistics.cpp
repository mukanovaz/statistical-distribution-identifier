#include "include/data.h"
#include<cmath>
#include <iostream>
#include <numeric>

#undef min
#undef max

#include <tbb/combinable.h>
#include <tbb/parallel_for.h>

namespace ppr
{
	// https://www.johndcook.com/blog/standard_deviation/
    class RunningStat
    {
        protected:
            double m_oldM;
            double m_newM;
            double  m_oldS;
            double m_newS;
            double m_sum;
            double m_sumAbs;
            double m_max;
            double m_min;
            int m_n;

        public:

            //RunningStat(){}

            RunningStat(const double first_x) 
                : m_n(1), m_oldM(first_x), m_newM(first_x), m_oldS(0.0), m_newS(0.0), m_sum(first_x), m_sumAbs(first_x), m_min(88888.0), m_max(0)
            {}

            void Clear()
            {
                m_n = 0;
            }

            void Push(double x)
            {
                m_n++;

                // See Knuth TAOCP vol 2, 3rd edition, page 232
                m_newM = m_oldM + (x - m_oldM) / m_n;
                m_newS = m_oldS + (x - m_oldM) * (x - m_newM);
                m_sum += x;

                m_min = x < m_min ? x : m_min;
                m_max = x > m_max ? x : m_max;

                // set up for next iteration
                m_oldM = m_newM;
                m_oldS = m_newS;
            }

            int NumDataValues() const
            {
                return m_n;
            }

            double Sum() const
            {
                return m_sum;
            }

            double SumAbs() const
            {
                return m_sumAbs;
            }

            double Get_Max() const
            {
                return m_max;
            }

            double Get_Min() const
            {
                return m_min;
            }

            double Mean() const
            {
                return (m_n > 0.0) ? m_newM : 0.0;
            }

            double Variance() const
            {
                return ((m_n > 1.0) ? m_newS / (m_n - 1.0) : 0.0);
            }

            double StandardDeviation() const
            {
                return sqrt(Variance());
            }
    };

    class Running_stat_parallel
    {       
        private:
            const double* m_data;
            SDataStat m_stat;
            const unsigned long long m_first_index;

        public:
            Running_stat_parallel(double* data, unsigned long long first_index) : m_data(data), m_first_index(first_index)
            {
                m_stat.n = 1;
                m_stat.sum = data[m_first_index];
                m_stat.min = 88888.0; // TODO: min
                m_stat.max = -88888.0;
                m_stat.isNegative = true;
            }

            Running_stat_parallel(Running_stat_parallel& x, tbb::split) : m_data(x.m_data), m_first_index(x.m_first_index)
            {
                m_stat.n = 1;
                m_stat.sum = x.m_data[m_first_index];
                m_stat.min = 88888.0; // TODO: min
                m_stat.max = -88888.0;
                m_stat.isNegative = true;
            }

            void operator()(const tbb::blocked_range<size_t>& r)
            {
                // Parameters 
                const double* t_data = m_data;
                SDataStat t_stat = m_stat;
                size_t begin = r.begin();
                size_t end = r.end();

                for (size_t i = begin; i != end; i++)
                {
                    double x = (double)t_data[i];

                    t_stat.n += 1;
                    t_stat.sum += x;
                    t_stat.isNegative = t_stat.isNegative || std::signbit(x);

                    t_stat.min = x < t_stat.min ? x : t_stat.min;
                    t_stat.max = x > t_stat.max ? x : t_stat.max;
                }
                m_stat = t_stat;
            }

            void join(const Running_stat_parallel& y)
            {
                m_stat.n += y.m_stat.n;
                m_stat.sum += y.m_stat.sum; 
                m_stat.min = m_stat.min < y.m_stat.min ? m_stat.min : y.m_stat.min;
                m_stat.max = m_stat.max > y.m_stat.max ? m_stat.max : y.m_stat.max;
                m_stat.isNegative = m_stat.isNegative || y.m_stat.isNegative;
            }

            unsigned long long NumDataValues() const
            {
                return m_stat.n;
            }

            double Sum() const
            {
                return m_stat.sum;
            }

            bool IsNegative() const
            {
                return m_stat.isNegative;
            }

            double Get_Max() const
            {
                return m_stat.max;
            }

            double Get_Min() const
            {
                return m_stat.min;
            }
    };
}

