#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <CL/cl.hpp>

namespace ppr
{
    constexpr int THREAD_PER_CORE = 1;
    const constexpr char* STAT_KERNEL = "D:/Study/ZCU/5.semestr/PPR/kiv-ppr/msvc/statistics_kernel.cl";
    const constexpr char* STAT_KERNEL_NAME = "Get_Data_Statistics";
    const constexpr char* HIST_KERNEL = "D:/Study/ZCU/5.semestr/PPR/kiv-ppr/msvc/histogram_kernel.cl";
    const constexpr char* HIST_KERNEL_NAME = "Get_Data_Histogram";
    //const constexpr DWORD MAX_FILE_SIZE_MEM = 1999962112;
    //const constexpr DWORD MAX_FILE_SIZE_MEM = 999948288;
    const constexpr DWORD MAX_FILE_SIZE_MEM = 499974144;
    const constexpr bool USE_OPTIMIZATION = true;

    enum class ERun_mode {
        SMP = 0,
        ALL = 1,
        SEQ = 2
    };

    enum EExitStatus : int {
        SUCCESS = 0,
        ARGS = 1,
        UNKNOWN = 2,
        FILE = 3,
        MAPPING = 4,
        STAT = 5
    };

    struct SConfig {
        const WCHAR* input_fn{};
        ERun_mode mode{};
        std::vector<std::string> cl_devices_name{};
        int thread_count = 0;
    };

    struct SOpenCLConfig {
        cl::Device device{};
        cl::Context context{};
        cl::Program program{};
        cl::Kernel kernel{};
        cl::Buffer out_sum_buf{};
        cl::CommandQueue queue{};
        size_t wg_size = 0;
        unsigned long data_count_for_gpu = 0;
        unsigned long data_count_for_cpu = 0;
        unsigned long wg_count = 0;
    };

    bool parse_args(int argc, char** argv, SConfig& config);

    void print_usage();

    void print_error(const char* message);
}
