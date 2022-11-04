#pragma once
#include <string>
#include <iostream>

namespace ppr
{
    enum class ERun_mode {
        SMP = 0,
        ALL = 1
    };

    enum EExitStatus : int {
        SUCCESS = 0,
        ARGS = 1,
        UNKNOWN = 2,
    };

    struct SConfig {
        std::string input_fn;
        ERun_mode mode;
        std::string cl_device_name;
    };

    bool parse_args(int argc, char** argv, SConfig& config);

    void print_usage();

    void print_error(const char* message);
}
