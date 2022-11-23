#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace ppr
{
    constexpr int THREAD_PER_CORE = 1;

    enum class ERun_mode {
        SMP = 0,
        ALL = 1
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
        const char* input_fn;
        ERun_mode mode;
        std::string cl_device_name;
        int thread_count;
    };

    bool parse_args(int argc, char** argv, SConfig& config);

    void print_usage();

    void print_error(const char* message);
}
