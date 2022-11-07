/*
* KIV/PPR 2022 refence distribution generator
*/
#include <random>
#include <iostream>
#include <fstream>

template <typename D, typename ...arg>
void Generate_Numbers(const char* dst, const size_t count, const arg... a) {
    std::ofstream output(dst, std::ios::out | std::ios::binary);
    if (output.is_open()) {

        std::random_device rd;
        D dis(a...);

        std::cout << "Generating the numbers... ";
        for (int n = 0; n < count; ++n) {
            const double num = dis(rd);
            output.write(reinterpret_cast<const char*>(&num), sizeof(num));
        }
        std::cout << "done.";
    }
    else {
        std::cout << "Cannot open the output file. Exiting..." << std::endl;        
    }
}



int emain(int argc, char* argv[]) {
       
    std::cout << "Usage: output_file distribution count min max" << std::endl;
    std::cout << "\toutput_file - a filename" << std::endl;
    std::cout << "\tDistribution: g/n - Gauss/normal; a=mean; b=stddev" << std::endl;
    std::cout << "\t              u - uniform; a=min; b=max" << std::endl;
    std::cout << "\t              e - exponential; a=lambda; b=0" << std::endl;
    std::cout << "\t              p - Poisson; a=mean; b=0" << std::endl;

    if (argc != 6) {
        std::cout << "Invalid number of parameters. Exiting..." << std::endl;
        return __LINE__;
    }


    const size_t count = std::strtoull(argv[3], nullptr, 0);
    const double a = std::strtod(argv[4], nullptr);
    const double b = std::strtod(argv[5], nullptr);

    
      
    switch (*argv[2]) {
        case 'g': [[fallthrough]];
        case 'n': Generate_Numbers<std::normal_distribution<>>(argv[1], count, a, b);
            break;

        case 'u': 
            {
                if (a >= b) {
                    std::cout << "Minimum must be less than maximum. Exiting..." << std::endl;
                    return __LINE__;
                }
                Generate_Numbers<std::uniform_real_distribution<>>(argv[1], count, a, b);
            }
            break;

        case 'e': Generate_Numbers<std::exponential_distribution<>>(argv[1], count, a);
            break;

        case 'p': Generate_Numbers<std::poisson_distribution<>>(argv[1], count, a);
            break;


        default:
            std::cout << "Unknown distribution! Exiting..." << std::endl;
            return __LINE__;
    }
            

    return 0;
}