#include <vector>
#include <fstream>
#include <numeric>
#include<cmath>
#include <iostream>
#include <random>
#include <string>
#include <iterator>
#include <algorithm>
# define M_PI           3.14159265358979323846

std::vector<double> load_doubles(std::string file_name, int floats_per_read) {
    size_t buffer_size = floats_per_read * sizeof(double);

    std::vector<double> doubles;

    // open file stream
    std::ifstream fin(file_name, std::ifstream::in | std::ifstream::binary);
    bool eof = false;

    // prepare a 8 byte buffer (for 64-bit float)
    std::vector<char> buffer(buffer_size, 0);

    while (!eof) {
        // read from the file stream
        fin.read(buffer.data(), buffer.size());

        // interpret the buffer as a double
        auto* read_doubles = (double*)buffer.data();

        for (int i = 0; i < fin.gcount() / sizeof(double); i++) {
            double d = read_doubles[i];
            doubles.push_back(d);
        }

        if (fin.gcount() < 1000) eof = true;
    }

    return doubles;
}

class RunningStat
{
    public:
        RunningStat() : m_n(0) {}

        void Clear()
        {
            m_n = 0;
        }

        void Push(double x)
        {
            m_n++;

            // See Knuth TAOCP vol 2, 3rd edition, page 232
            if (m_n == 1)
            {
                m_oldM = m_newM = x;
                m_oldS = 0.0;
            }
            else
            {
                m_newM = m_oldM + (x - m_oldM) / m_n;
                m_newS = m_oldS + (x - m_oldM) * (x - m_newM);

                // set up for next iteration
                m_oldM = m_newM;
                m_oldS = m_newS;
            }
        }

        int NumDataValues() const
        {
            return m_n;
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

    private:
        int m_n;
        double m_oldM, m_newM, m_oldS, m_newS;
};



double LogFactorial(int n)
{
    if (n < 0)
    {
        return 0.0;
    }
    else if (n > 254)
    {
        double x = n + 1;
        return (x - 0.5) * log(x) - x + 0.5 * log(2 * M_PI) + 1.0 / (12.0 * x);
    }
}

int nmain(int argc, char* argv[])
{
    std::string paths[4]
        = { "D:\\Study\\ZCU\\5.semestr\\PPR\\kiv-ppr\\referencni_rozdeleni\\exp", 
        "D:\\Study\\ZCU\\5.semestr\\PPR\\kiv-ppr\\referencni_rozdeleni\\gauss", 
        "D:\\Study\\ZCU\\5.semestr\\PPR\\kiv-ppr\\referencni_rozdeleni\\poisson", 
        "D:\\Study\\ZCU\\5.semestr\\PPR\\kiv-ppr\\referencni_rozdeleni\\uniform" };

    for (int i = 0; i < 4; i++)
    {
        RunningStat rs;
        std::vector<double> doubles = load_doubles(paths[i], 500);
        double doubles_size = doubles.size();
        int n = 323;
        std::vector<double> rand_doubles;

        // Random samlpe
        std::sample(doubles.begin(), doubles.end(), std::back_inserter(rand_doubles),
            n, std::mt19937{ std::random_device{}() });

        for (int i = 0; i < n; i++) {
            rs.Push(doubles[i]);
        }

        double mean = rs.Mean();
        double variance = rs.Variance();
        double stdev = rs.StandardDeviation();

        // ================ [Gauss maximum likelihood estimators]
        double sum_of_x = 0.0;
        for (int i = 0; i < n; i++) {
            sum_of_x += doubles[i];
        }

        double gauss_mean = (static_cast<double>(1) / static_cast<double>(n)) * sum_of_x;
        double gauss_variance_1 = 0.0;

        for (int i = 0; i < n; i++) {
            gauss_variance_1 += pow(doubles[i] - gauss_mean, 2);
        }

        double gauss_variance = (static_cast<double>(1) / static_cast<double>(n)) * gauss_variance_1;

        // ================ [Exponential maximum likelihood estimators]
        double exp_lambda = static_cast<double>(n) / sum_of_x;
        

        // ================ [Poisson]
        long poisson_lambda = (static_cast<double>(1) / static_cast<double>(n)) * sum_of_x;
       
        // ================ [Uniform]
        auto a = std::min_element(doubles.begin(), doubles.end());
        auto b = std::max_element(doubles.begin(), doubles.end());


       // Generate 
       bool exp_is_lambda;
       std::default_random_engine generator;


       double gauss_rss = 0.0;
       std::normal_distribution<double> normal_distribution(gauss_mean, stdev);
       for (int i = 0; i < n; i++) {
           gauss_rss += pow(doubles[i] - normal_distribution(generator), 2);
       }

       double exp_rss = 0.0;
       if (exp_lambda > 0)
       {
           std::exponential_distribution<double> exponential_distribution(exp_lambda);
           for (int i = 0; i < n; i++) {
               exp_rss += pow(doubles[i] - exponential_distribution(generator), 2);
           }
       }
       else
       {
           exp_rss = 88888.8;
       }

       
       double uniform_rss = 0.0;
       std::uniform_real_distribution<double> uniform_real_distribution(*a, *b);
       for (int i = 0; i < n; i++) {
           uniform_rss += pow(doubles[i] - uniform_real_distribution(generator), 2);
       }
       
       double poisson_rss = 0.0;
       if (poisson_lambda > 0)
       {
           std::poisson_distribution<int> poisson_distribution(poisson_lambda);
           for (int i = 0; i < n; i++) {
               poisson_rss += pow(doubles[i] - poisson_distribution(generator), 2);
           }
       }
       else
       {
           poisson_rss = 88888.8;
       }
      

       
       std::cout << "Gauss: " << gauss_rss << std::endl;
       std::cout << "Exponential: " << exp_rss << std::endl;
       std::cout << "Poisson: " << poisson_rss << std::endl;
       std::cout << "Uniform: " << uniform_rss << std::endl;

       double max = std::min({ gauss_rss, exp_rss, poisson_rss, uniform_rss });

       if (gauss_rss == max)
       {
           std::cout << "This is Gauss" << std::endl;
       }
       else if (exp_rss == max)
       {
           std::cout << "This is Exponential" << std::endl;
       }
       else if (poisson_rss == max)
       {
           std::cout << "This is Poisson" <<  std::endl;
       }
       else if (uniform_rss == max)
       {
           std::cout << "This is Uniform" << std::endl;
       }

       std::cout << "==========================================" << std::endl;
    }


    getchar();
    return 0;
}

