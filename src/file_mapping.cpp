#include "include/file_mapping.h"

namespace ppr
{
    File_mapping::File_mapping(const WCHAR* filename)
        : m_filename(filename), m_file(INVALID_HANDLE_VALUE), m_mapping(INVALID_HANDLE_VALUE), m_data(NULL)
    {
        bool res_cf = create_file_n();
        if (!res_cf)
        {
            return;
        }

        // Get file size and number of doubles inside
        LARGE_INTEGER file_size = { 0 };
        ::GetFileSizeEx(m_file, &file_size);
        m_fileLen = static_cast<unsigned long long>(file_size.QuadPart);
        m_size = m_fileLen / sizeof(double);

        // Map a file
        bool res_mf = map_file();
        if (!res_mf)
        {
            ppr::print_error("Cannot create a mapping");
            CloseHandle(m_file);
            return;
        }

        // Create map view
        view();
    }

    File_mapping::File_mapping(SConfig& config)
        : m_filename(config.input_fn), m_file(INVALID_HANDLE_VALUE), m_mapping(INVALID_HANDLE_VALUE), m_data(NULL)
    {
        bool res_cf = create_file_n();
        if (!res_cf)
        {
            return;
        }

        // Offsets must be a multiple of the system's allocation granularity.  We
        // guarantee this by making our view size equal to the allocation granularity.
        SYSTEM_INFO sysinfo = { 0 };
        ::GetSystemInfo(&sysinfo);

        m_allocationGranularity = sysinfo.dwAllocationGranularity;

        // Get file size and number of doubles inside
        LARGE_INTEGER file_size = { 0 };
        ::GetFileSizeEx(m_file, &file_size);
        m_fileLen = static_cast<unsigned long long>(file_size.QuadPart);
        m_size = m_fileLen / sizeof(double);
        CloseHandle(m_file);

        // For OpecCl devices we are choosing smaller parts
        if (config.mode == ERun_mode::ALL || m_fileLen < MAX_FILE_SIZE_MEM_500mb)
        {
            m_scale = static_cast<double>(MAX_FILE_SIZE_MEM_500mb) / sysinfo.dwAllocationGranularity;
        }
    }

    bool File_mapping::create_file_n()
    {
        m_file = CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

        if (m_file == INVALID_HANDLE_VALUE)
        {
            return false;
        };

        return true;
    }

    bool File_mapping::map_file() {

        m_mapping = CreateFileMapping(m_file, 0, PAGE_READONLY, 0, 0, 0);
        if (m_mapping == 0)
        {
            CloseHandle(m_file);
            return false;
        };

        return true;
    }


    void File_mapping::view()
    {
        m_data = (double*)MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
    }

    double* File_mapping::get_data() const
    {
        return m_data;
    }

    const unsigned int File_mapping::get_file_len() const
    {
        return m_fileLen;
    }

    const unsigned int File_mapping::get_count() const
    {
        return m_size;
    }

    void File_mapping::unmap_file()
    {
        UnmapViewOfFile(m_data);
        CloseHandle(m_mapping);
        CloseHandle(m_file);
    }

    const DWORD File_mapping::get_granularity() const
    {
        return m_allocationGranularity;
    }

    void File_mapping::read_in_one_chunk_cpu(
        SHistogram& hist,
        SConfig& config,
        SOpenCLConfig& opencl,
        SDataStat& stat,
        EIteration iteration,
        std::vector<int>& histogram)
    {
        DWORD granulatity = m_allocationGranularity * m_scale;

        HANDLE hfile = ::CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hfile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER file_size = { 0 };
            ::GetFileSizeEx(hfile, &file_size);
            const unsigned long long cbFile =
                static_cast<unsigned long long>(file_size.QuadPart);

            // Create a file mapping
            HANDLE hmap = ::CreateFileMappingW(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (hmap != NULL) {
                // Create a chunk
                double* pView = static_cast<double*>(
                    ::MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0));

                if (pView != NULL) {
                    unsigned int data_in_chunk = cbFile / sizeof(double);

                    // Set computing limits
                    opencl.data_count_for_cpu = data_in_chunk / config.thread_count;

                    if (iteration == EIteration::STAT)
                    {
                        std::vector<std::future<SDataStat>> workers(config.thread_count);
                        for (int i = 0; i < config.thread_count; i++)
                        {
                            ppr::parallel::Stat_processing_unit unit(config, opencl);
                            workers[i] = std::async(std::launch::async, &ppr::parallel::Stat_processing_unit::run_on_CPU, unit, pView + (opencl.data_count_for_cpu * i), opencl.data_count_for_cpu);
                        }

                        // Agregate results results
                        for (auto& worker : workers)
                        {
                            SDataStat local_stat = worker.get();
                            stat.sum += local_stat.sum;
                            stat.n += local_stat.n;
                            stat.max = std::max({ stat.max, local_stat.max });
                            stat.min = std::min({ stat.min, local_stat.min });
                        }
                    }
                    else
                    {
                        std::vector<std::future<std::tuple<std::vector<int>, double>>> workers(config.thread_count);

                        for (int i = 0; i < config.thread_count; i++)
                        {
                            ppr::parallel::Hist_processing_unit unit(hist, config, opencl, stat);
                            workers[i] = std::async(std::launch::async, &ppr::parallel::Hist_processing_unit::run_on_CPU, unit, pView + (opencl.data_count_for_cpu * i), opencl.data_count_for_cpu);
                        }

                        // Agregate results results
                        for (auto& worker : workers)
                        {
                            auto [vector, variance] = worker.get();
                            stat.variance += variance;
                            std::transform(histogram.begin(), histogram.end(), vector.begin(), histogram.begin(), std::plus<int>());
                        }
                    }
                    UnmapViewOfFile(pView);
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
    }

    void File_mapping::read_in_chunks(
        SHistogram& hist,
        SConfig& config,
        SOpenCLConfig& opencl,
        SDataStat& stat,
        tbb::task_arena& arena,
        std::vector<int>& histogram,
        void (*process_chunk) (SHistogram& hist, SConfig&, SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned int, double*, std::vector<int>&))
    {
        DWORD granulatity = m_allocationGranularity * m_scale;

        // Crate a file
        HANDLE hfile = ::CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hfile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER file_size = { 0 };
            ::GetFileSizeEx(hfile, &file_size);
            const unsigned long long cbFile =
                static_cast<unsigned long long>(file_size.QuadPart);

            // Create a file mapping
            HANDLE hmap = ::CreateFileMappingW(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (hmap != NULL) {
                for (unsigned long long offset = 0; offset < cbFile; offset += granulatity) {
                    // Get chunk limits
                    DWORD high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
                    DWORD low = static_cast<DWORD>(offset & 0xFFFFFFFFul);
                    // The last view may be shorter.
                    if (offset + granulatity > cbFile) {
                        granulatity = static_cast<int>(cbFile - offset);
                    }

                    // Map one chunk
                    double* pView = static_cast<double*>(
                        ::MapViewOfFile(hmap, FILE_MAP_READ, high, low, granulatity));

                    if (pView != NULL) {
                        unsigned int data_in_chunk = granulatity / sizeof(double);

                        // Set opencl computing limits
                        if (opencl.wg_size != 0)
                        {
                            // Get number of data, which we want to process on GPU
                            opencl.wg_count = data_in_chunk / opencl.wg_size;
                            opencl.data_count_for_gpu = data_in_chunk - (data_in_chunk % opencl.wg_size);

                            // The rest of the data we will process on CPU
                            opencl.data_count_for_cpu = opencl.data_count_for_gpu + 1;
                        }
                        else
                        {
                            opencl.data_count_for_cpu = 0;
                        }

                        // Run
                        process_chunk(hist, config, opencl, stat, arena, data_in_chunk, pView, histogram);

                        UnmapViewOfFile(pView);
                    }
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
    }



}