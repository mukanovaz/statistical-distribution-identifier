#include<cmath>

#undef min
#undef max

#include <tbb/combinable.h>
#include <tbb/parallel_for.h>
#include "../data.h"
#include <iostream>

namespace ppr
{
	// https://www.johndcook.com/blog/standard_deviation/
    class RunningStat
    {
        protected:
            int m_n;
            double m_oldM, m_newM, m_oldS, m_newS, m_sum, m_sumAbs, m_max, m_min;

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
                m_sumAbs += fabs(x);

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

    class RunningStatParallel
    {       
        private:
            SStat m_stat;
            const double* m_data;
            const int m_first_index;

        public:
            RunningStatParallel(double* data, int first_index) : m_data(data), m_first_index(first_index)
            {
                m_stat.n = 1;
                m_stat.oldM = data[m_first_index];
                m_stat.newM = data[m_first_index];
                m_stat.oldS = 0.0;
                m_stat.newS = 0.0;
                m_stat.sum = data[m_first_index];
                m_stat.sumAbs = data[m_first_index];
                m_stat.min = 88888.0; // TODO: min
                m_stat.max = 0.0;
            }

            RunningStatParallel(RunningStatParallel& x, tbb::split) : m_data(x.m_data), m_first_index(x.m_first_index)
            {
                m_stat.n = 1;
                m_stat.oldM = x.m_data[m_first_index];
                m_stat.newM = x.m_data[m_first_index];
                m_stat.oldS = 0.0;
                m_stat.newS = 0.0;
                m_stat.sum = 0.0;
                m_stat.sumAbs = 0.0;
                m_stat.min = 88888.0; // TODO: min
                m_stat.max = 0.0;
            }

            void operator()(const tbb::blocked_range<size_t>& r)
            {
                // Parameters 
                const double* t_data = m_data;
                SStat t_stat = m_stat;

                for (size_t i = r.begin(); i != r.end(); i++)
                {
                    //std::cout << i << std::endl;
                    double x = (double)t_data[i];

                    t_stat.n++;

                    // See Knuth TAOCP vol 2, 3rd edition, page 232
                    t_stat.newM = t_stat.oldM + (x - t_stat.oldM) / t_stat.n;
                    t_stat.newS = t_stat.oldS + (x - t_stat.oldM) * (x - t_stat.newM);
                    t_stat.sum += x;
                    t_stat.sumAbs += fabs(x);

                    t_stat.min = x < t_stat.min ? x : t_stat.min;
                    t_stat.max = x > t_stat.max ? x : t_stat.max;

                    // set up for next iteration
                    t_stat.oldM = t_stat.newM;
                    t_stat.oldS = t_stat.newS;
                }
                m_stat = t_stat;
            }

            void join(const RunningStatParallel& y)
            {
                m_stat.n += y.m_stat.n;
                m_stat.newM += y.m_stat.newM;
                m_stat.newS += y.m_stat.newS;
                m_stat.sum += y.m_stat.sum; 
                m_stat.sumAbs += y.m_stat.sumAbs;
                m_stat.min = m_stat.min < y.m_stat.min ? m_stat.min : y.m_stat.min;
                m_stat.max = m_stat.max > y.m_stat.max ? m_stat.max : y.m_stat.max;
            }

            int NumDataValues() const
            {
                return m_stat.n;
            }

            double Sum() const
            {
                return m_stat.sum;
            }

            double SumAbs() const
            {
                return m_stat.sumAbs;
            }

            double Get_Max() const
            {
                return m_stat.max;
            }

            double Get_Min() const
            {
                return m_stat.min;
            }

            double Mean() const
            {
                return (m_stat.n > 0.0) ? m_stat.newM : 0.0;
            }

            double Variance() const
            {
                return ((m_stat.n > 1.0) ? m_stat.newS / (m_stat.n - 1.0) : 0.0);
            }

            double StandardDeviation() const
            {
                return sqrt(Variance());
            }
    };
}

