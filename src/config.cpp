#include "include/config.h"
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

		int man_argc = 0;

		// Check if file exist
		config.input_fn = Char2Wchar(argv[1]);

		std::string file_name = argv[1];
		man_argc++;
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
			man_argc++;
			config.mode = ERun_mode::SMP;
		}
		else if (std::strncmp("seq", argv[2], 3) == 0)
		{
			man_argc++;
			config.mode = ERun_mode::SEQ;
		}
		else if (std::strncmp("all", argv[2], 3) == 0)
		{
			config.mode = ERun_mode::ALL;
			man_argc += 2;

			// Save devices
			while (man_argc < argc && argv[man_argc][0] != '-')
			{
				config.cl_devices_name.push_back(argv[man_argc]);
				man_argc++;
			}
		}
		else
		{
			print_error("unknown mode!");
			print_usage();
			return false;
		}

		// Get optional
		for (int i = man_argc; i < argc; i += 2)
		{
			if (i + 1 >= argc)
			{
				print_error("wrong number of arguments!");
			}

			// use optimalization
			if (std::strncmp("-o", argv[i], 2) == 0)
			{
				int opt = 0;
				if (sscanf_s(argv[i + 1], "%d", &opt) != 1)
				{
					print_error("wrong argument type!");
					print_usage();
					return false;
				}
				if (opt != 0 && opt != 1)
				{
					print_error("Wrong argument type! Should be '1' or '0'");
					print_usage();
					return false;
				}

				config.use_optimalization = opt == 1;
			}

			// thread per code
			if (std::strncmp("-t", argv[i], 2) == 0)
			{
				int tc = 0;
				if (sscanf_s(argv[i + 1], "%d", &tc) != 1)
				{
					print_error("wrong argument type!");
					print_usage();
					return false;
				}

				config.thread_count = tc;
			}

			// watchdog interval
			if (std::strncmp("-w", argv[i], 2) == 0)
			{
				int wi = 0;
				if (sscanf_s(argv[i + 1], "%d", &wi) != 1)
				{
					print_error("wrong argument type!");
					print_usage();
					return false;
				}

				config.watchdog_interval = wi;
			}

		}

		// Find number available of threads
		config.thread_count =  static_cast<int>(std::thread::hardware_concurrency()) * config.thread_per_core;

		return true;
	}

	void print_usage()
	{
		std::cout << "+-------------------------------------------------------+" << std::endl;
		std::cout << "|\t\tPROBABILITY DISTRIBUTION FITTING\t|" << std::endl;
		std::cout << "+-------------------------------------------------------+" << std::endl;
		std::cout << "| * path for input file\t\t\t\t\t|" << std::endl;
		std::cout << "| * run mode [all (SMP and OpenCL) / SMP]\t\t|" << std::endl;
		std::cout << "| * opencl devices name\t\t\t\t\t|" << std::endl;
		std::cout << "| \t\t=== [optional] ===\t\t\t|" << std::endl;
		std::cout << "| * -o\t\tuse optimalization [1/0] ('1' default)\t|" << std::endl;
		std::cout << "| * -t\t\tthread per code [int] ('1' default)\t|" << std::endl;
		std::cout << "| * -w\t\twatchdog interval [sec] ('2' default)\t|" << std::endl;
		std::cout << "| * -st\t\tstatistics timeout [sec] ('5' default)\t|" << std::endl;
		std::cout << "+-------------------------------------------------------+" << std::endl;
	}

	void print_error(const char* message)
	{
		std::cerr << "> [ERROR]: " << message << std::endl;
	}

	void print_error(const std::string message)
	{
		std::cerr << "> [ERROR]: " << message << std::endl;
	}
}
