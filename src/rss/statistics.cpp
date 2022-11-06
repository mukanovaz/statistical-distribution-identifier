#include<cmath>

namespace ppr
{
	// https://www.johndcook.com/blog/standard_deviation/
    class RunningStat
    {
        private:
            int m_n;
            double m_oldM, m_newM, m_oldS, m_newS, sum, sum2, max, min;

        public:
            RunningStat(const double first_x) 
                : m_n(1), m_oldM(first_x), m_newM(first_x), m_oldS(0.0), m_newS(0.0), sum(0.0), sum2(0.0), min(88888.0), max(0)
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
                sum += x;
                sum2 += abs(x);

                min = x < min ? x : min;
                max = x > max ? x : max;

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
                return sum;
            }

            double Get_Max() const
            {
                return max;
            }

            double Get_Min() const
            {
                return min;
            }

            double Mean() const
            {
                return (m_n > 0) ? m_newM : 0.0;
            }

            double Variance() const
            {
                return ((m_n > 1) ? m_newS / (m_n - 1) : 0.0);
            }

            double StandardDeviation() const
            {
                return sqrt(Variance());
            }
    };
}

