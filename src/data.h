#pragma once

enum class EDistribution {
    GAUSS = 0,
    EXP = 1,
    POISSON = 2,
    UNIFORM = 3
};

struct SResult
{
    EDistribution dist;     // Result distribution
    int status;
    double gauss_mean;
    double gauss_variance;
    double exp_lambda;
    long poisson_lambda;
    double uniform_a;
    double uniform_b;
    double gauss_rss;
    double exp_rss;
    double poisson_rss;
    double uniform_rss;
    bool isNegative;
    bool isInteger;

    static SResult error_res(int exit_status)
    {
        SResult result;
        result.status = exit_status;
        return result;
    }
};