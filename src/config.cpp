#include "config.h"
#include <thread>

namespace ppr
{
	const WCHAR* Char2Wchar(char const* c)
	{
		size_t size = strlen(c) + 1;
		WCHAR* wc = new WCHAR[size];

		size_t outSize;
		mbstowcs_s(&outSize, wc, size, c, size - 1);
		return wc;
	}

	char asciitolower(char in) {
		if (in <= 'Z' && in >= 'A')
			return in - ('Z' - 'z');
		return in;
	}

	bool parse_args(int argc, char** argv, SConfig& config)
	{
		if (argc < 4)
		{
			print_error("wrong number of arguments!");
			print_usage();
			return false;
		}

		// Check if file exist
		config.input_fn = Char2Wchar(argv[1]);

		std::string file_name = argv[1];
		std::ifstream file(file_name, std::ios::binary);

		bool isGood = file.good();
		file.close();

		if (!isGood)
		{
			print_error("input file is not exist!");
			print_usage();
			return false;
		}

		// Get Mode type
		if (std::strncmp("smp", argv[2], 3) == 0)
		{
			config.mode = ERun_mode::SMP;
		}
		else if (std::strncmp("all", argv[2], 3) == 0)
		{
			config.mode = ERun_mode::ALL;
			// TODO: add more devices
			//config.cl_devices_name = argv[3];
		}
		else
		{
			print_error("unknown mode!");
			return false;
		}

		// Find number available of threads
		config.thread_count = THREAD_PER_CORE * std::thread::hardware_concurrency();

		return true;
	}

	void print_usage()
	{
		std::cout << "+-------------------------------------------------------+" << std::endl;
		std::cout << "|\t\tPROBABILITY DISTRIBUTION FITTING\t|" << std::endl;
		std::cout << "+-------------------------------------------------------+" << std::endl;
		std::cout << "|\t* path for input file\t\t\t\t|" << std::endl;
		std::cout << "|\t* processor type [all (SMP and OpenCL) / SMP]\t|" << std::endl;
		std::cout << "|\t* opencl devices name\t\t\t\t|" << std::endl;
		std::cout << "+-------------------------------------------------------+" << std::endl;
	}

	void print_error(const char* message)
	{
		std::cerr << "[ERROR]: " << message << std::endl;
	}
}
