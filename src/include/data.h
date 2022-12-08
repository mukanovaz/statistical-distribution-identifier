#pragma once
#include<vector>

/// <summary>
/// Enum with iterations
/// </summary>
enum class EIteration {
    STAT = 0,
    HIST = 1
};

/// <summary>
/// Enum with all existing distributions
/// </summary>
enum class EDistribution {
    GAUSS = 0,
    EXP = 1,
    POISSON = 2,
    UNIFORM = 3
};

/// <summary>
/// HIstogram configuration structure
/// </summary>
struct SHistogram
{
    double binSize = 0.0;
    double max = 0.0;
    double min = 0.0;
    double scaleFactor = 0.0;
    int binCount = 0;
};

/// <summary>
/// Statistics collected from data. (Using only for sequential computing and TBB algoorithms)
/// </summary>
struct SStat
{
    double oldM = 0.0;
    double newM = 0.0;
    double oldS = 0.0;
    double newS = 0.0;
    double sum = 0.0;
    double sumAbs = 0.0;
    double max = 0.0;
    double min = 0.0;
    unsigned int n = 0;
    bool isNegative = false;
};

/// <summary>
/// Statistics collected from data
/// </summary>
struct SDataStat
{
    double sum = 0.0;
    double max = 0.0;
    double min = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    bool isNegative = 0;
    unsigned long long n = 0;
};

/// <summary>
/// Structure with computing results
/// </summary>
struct SResult
{
    EDistribution dist{};     // Result distribution
    double gauss_mean = 0.0;
    double gauss_variance = 0.0;
    double gauss_stdev = 0.0;
    double exp_lambda = 0.0;
    double poisson_lambda = 0;
    double uniform_a = 0.0;
    double uniform_b = 0.0;
    double gauss_rss = 0.0;
    double exp_rss = 0.0;
    double poisson_rss = 0.0;
    double uniform_rss = 0.0;
    double total_stat_time = 0.0;
    double total_hist_time = 0.0;
    double total_rss_time = 0.0;
    double total_time = 0.0;
    int status = 0;
    bool isNegative = 0;
    bool isInteger = 0.0;

    static SResult error_res(int exit_status)
    {
        SResult result;
        result.status = exit_status;
        return result;
    }
};