#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Windows.h>

#include "data.h"

namespace ppr
{
    const WCHAR* char2wchar(char const* c);

    /// <summary>
    /// Constant for definition how many thread should run on 1 CPU core
    /// </summary>
    const constexpr int THREAD_PER_CORE = 1;
    /// <summary>
    /// File path for opencl statistics kernel
    /// </summary>
    const constexpr char* STAT_KERNEL = "cl\\statistics_kernel.cl"; 
    /// <summary>
    /// Name of opencl statistics kernel
    /// </summary>
    const constexpr char* STAT_KERNEL_NAME = "Get_Data_Statistics";
    /// <summary>
    /// File path for opencl histogram kernel
    /// </summary>
    const constexpr char* HIST_KERNEL = "cl\\histogram_kernel.cl";
    /// <summary>
    /// Name of opencl histogram kernel
    /// </summary>
    const constexpr char* HIST_KERNEL_NAME = "Get_Data_Histogram";
    /// <summary>
    /// 2 Gigabytes in bytes aligned for allocation granularity on current system
    /// </summary>
    const constexpr long MAX_FILE_SIZE_MEM_2gb = 1999962112;
    /// <summary>
    /// 1 Gigabyte in bytes aligned for allocation granularity on current system
    /// </summary>
    const constexpr long MAX_FILE_SIZE_MEM_1gb = 999948288;
    /// <summary>
    /// 0.5 Gigabyte in bytes aligned for allocation granularity on current system
    /// </summary>
    const constexpr long MAX_FILE_SIZE_MEM_500mb = 499974144;
    /// <summary>
    /// Watchdog default interval
    /// </summary>
    const constexpr int WATCHDOG_INTERVAL_SEC = 2;
    /// <summary>
    /// Timeout for get statistics from one chunk of file. Using for changing allocation granularity scale.
    /// </summary>
    const constexpr int STAT_TIMEOUT_SEC = 5;
    /// <summary>
    /// Default optimalization setting
    /// </summary>
    const constexpr bool USE_OPTIMIZATION = true;

    /// <summary>
    /// Enum class for program mode definition
    /// </summary>
    enum class ERun_mode {
        SMP = 0,
        ALL = 1,
        SEQ = 2,
        CL = 3

    };

    inline const char* print_mode(ERun_mode v)
    {
        switch (v)
        {
            case ERun_mode::SMP:    
                return "SMP";
            case ERun_mode::ALL:    
                return "ALL";
            case ERun_mode::SEQ:    
                return "Sequential";
            case ERun_mode::CL: 
                return "On selected devices";

            default:      return "[Unknown OS_type]";
        }
    }

    /// <summary>
    /// Enum for exit codes definition
    /// </summary>
    enum EExitStatus : int {
        SUCCESS = 0,
        ARGS = 1,
        UNKNOWN = 2,
        FILE = 3,
        MAPPING = 4,
        STAT = 5,
        HIST = 6,
        WD_STAT_WRONG_N = 7,
        WD_STAT_MIN_MAX = 8,
        WD_HIST_BIN_COUNT = 9,
        WD_HIST_SIZE = 10,
        WD_HIST_ALL_ZERO = 11,
        WD_DHIST_ALL_ZERO = 12
    };

    /// <summary>
    /// Structure of program configuration
    /// </summary>
    struct SConfig {
        const char* input_fn{};                         // Input file name
        ERun_mode mode{};                               // Program running mode
        std::vector<std::string> cl_devices_name{};     // OpenCl Devices from user input
        int thread_count = 0;                           // System max thread count
        long long watchdog_interval = WATCHDOG_INTERVAL_SEC;
        long long stat_timeout = STAT_TIMEOUT_SEC;
        bool use_optimalization = USE_OPTIMIZATION;
        int thread_per_core = THREAD_PER_CORE;
    };

    /// <summary>
    /// Parse user arguments and save it into SConfig structure
    /// </summary>
    /// <param name="argc">Number of arguments</param>
    /// <param name="argv">Arguments</param>
    /// <param name="config">Program configuration structure</param>
    /// <returns>Is success</returns>
    bool parse_args(int argc, char** argv, SConfig& config);

    /// <summary>
    /// Print usage 
    /// </summary>
    void print_usage();

    /// <summary>
    /// Print error message
    /// </summary>
    /// <param name="message">message</param>
    void print_error(const char* message);

    /// <summary>
    /// Print error message
    /// </summary>
    /// <param name="message">message</param>
    void print_error(const std::string message);

    /// <summary>
    /// Pring final statistics
    /// </summary>
    /// <param name="stat">Statistics structure</param>
    /// <param name="res">Result structure</param>
    void print_stat(SDataStat& stat, SResult& res);
}
