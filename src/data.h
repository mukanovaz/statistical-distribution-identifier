#pragma once
#include<vector>

enum class EDistribution {
    GAUSS = 0,
    EXP = 1,
    POISSON = 2,
    UNIFORM = 3
};


struct SHistogram
{
    double binSize = 0.0;
    double max = 0.0;
    double min = 0.0;
    double size = 0.0;
    double scaleFactor = 0.0;
};

struct SStat
{
    int n = 0;
    double oldM = 0.0;
    double newM = 0.0;
    double oldS = 0.0;
    double newS = 0.0;
    double sum = 0.0;
    double sumAbs = 0.0;
    double max = 0.0;
    double min = 0.0;
};

struct SResult
{
    EDistribution dist;     // Result distribution
    int status;
    double gauss_mean = 0.0;
    double gauss_variance = 0.0;
    double exp_lambda = 0.0;
    long poisson_lambda = 0.0;
    double uniform_a = 0.0;
    double uniform_b = 0.0;
    double gauss_rss = 0.0;
    double exp_rss = 0.0;
    double poisson_rss = 0.0;
    double uniform_rss = 0.0;
    bool isNegative = 0.0;
    bool isInteger = 0.0;

    static SResult error_res(int exit_status)
    {
        SResult result;
        result.status = exit_status;
        return result;
    }
};