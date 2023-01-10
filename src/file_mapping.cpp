#include "include/file_mapping.h"

namespace ppr
{
    const WCHAR* char2wchar(char const* c)
    {
        size_t size = strlen(c) + 1;
        WCHAR* wc = new WCHAR[size];

        size_t outSize;
        mbstowcs_s(&outSize, wc, size, c, size - 1);
        return wc;
    }

    File_mapping::File_mapping(const char* filename)
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
        m_size = static_cast<DWORD>(m_fileLen / sizeof(double));

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
        m_size = static_cast<DWORD>(m_fileLen / sizeof(double));
        CloseHandle(m_file);

        // For OpecCl devices we are choosing smaller parts
        if (config.mode == ERun_mode::ALL || config.mode == ERun_mode::CL || m_fileLen < MAX_FILE_SIZE_MEM_500mb)
        {
            m_scale = MAX_FILE_SIZE_MEM_500mb / sysinfo.dwAllocationGranularity;
        }
        // For big files we are choosing bigger parts
        else if (m_fileLen > MAX_FILE_SIZE_MEM_2gb)
        {
            m_scale = MAX_FILE_SIZE_MEM_2gb / sysinfo.dwAllocationGranularity;
        }
        else
        {
            m_scale = MAX_FILE_SIZE_MEM_1gb / sysinfo.dwAllocationGranularity;
        }
    }

    bool File_mapping::create_file_n()
    {
        const WCHAR* filename = char2wchar(m_filename);
        m_file = CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

        if (m_file == INVALID_HANDLE_VALUE)
        {
            delete filename;
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

    const unsigned long long File_mapping::get_file_len() const
    {
        return m_fileLen;
    }

    const long File_mapping::get_count() const
    {
        return m_size;
    }

    void File_mapping::unmap_file()
    {
        UnmapViewOfFile(m_data);
        CloseHandle(m_mapping);
        CloseHandle(m_file);
    }

    const long File_mapping::get_granularity() const
    {
        return m_allocationGranularity;
    }

    void File_mapping::read_in_one_chunk_cpu(
        SHistogram& hist,
        SConfig& config,
        ppr::gpu::SOpenCLConfig& opencl,
        SDataStat& stat,
        EIteration iteration,
        std::vector<int>& histogram)
    {
        DWORD granulatity = m_allocationGranularity * m_scale;

        const WCHAR* filename = char2wchar(m_filename);
        HANDLE hfile = ::CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ,
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
                    unsigned long long data_in_chunk = cbFile / sizeof(double);

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
                        // Histogram vector + variance
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
        delete filename;
    }

    void File_mapping::read_in_chunks_gpu(
        SHistogram& hist,
        SConfig& config,
        SDataStat& stat,
        EIteration iteration,
        std::vector<int>& histogram)
    {

        // Find all devices on all platforms
        std::vector<cl::Device> devices;
        ppr::gpu::find_opencl_devices(devices, config.cl_devices_name);
        DWORD granulatity = m_allocationGranularity * m_scale;

        // Create a file
        const WCHAR* filename = char2wchar(m_filename);
        HANDLE hfile = ::CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ,
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
                        unsigned long long data_in_chunk = granulatity / sizeof(double);

                        if (iteration == EIteration::STAT)
                        {
                            std::vector<std::future<SDataStat>> workers(devices.size());

                            for (int i = 0; i < devices.size(); i++)
                            {
                                ppr::gpu::SOpenCLConfig opencl;
                                opencl.device = devices[i];
                                ppr::gpu::set_kernel_program(opencl, STAT_KERNEL, STAT_KERNEL_NAME);

                                // Set opencl computing limits
                                if (opencl.wg_size != 0)
                                {
                                    opencl.data_count_for_gpu = (data_in_chunk - (data_in_chunk % opencl.wg_size)) / devices.size();
                                }

                                ppr::parallel::Stat_processing_unit unit(config, opencl);
                                workers[i] = std::async(std::launch::async, &ppr::parallel::Stat_processing_unit::run_on_GPU, unit, pView, opencl.data_count_for_gpu * i,
                                    (opencl.data_count_for_gpu * i) + opencl.data_count_for_gpu);
                            }

                            // Collect results
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
                            // Histogram vector + variance
                            std::vector<std::future<std::tuple<std::vector<int>, double>>> workers(devices.size());

                            for (int i = 0; i < devices.size(); i++)
                            {
                                ppr::gpu::SOpenCLConfig opencl;
                                opencl.device = devices[i];
                                ppr::gpu::set_kernel_program(opencl, HIST_KERNEL, HIST_KERNEL_NAME);

                                // Set opencl computing limits
                                if (opencl.wg_size != 0)
                                {
                                    opencl.data_count_for_gpu = (data_in_chunk - (data_in_chunk % opencl.wg_size)) / devices.size();
                                }

                                ppr::parallel::Hist_processing_unit unit(hist, config, opencl, stat);
                                workers[i] = std::async(std::launch::async, &ppr::parallel::Hist_processing_unit::run_on_GPU, unit, pView, opencl.data_count_for_gpu * i,
                                    (opencl.data_count_for_gpu * i) + opencl.data_count_for_gpu);
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
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
        delete filename;
    }

    void File_mapping::read_in_chunks_tbb(
        SHistogram& hist,
        SConfig& config,
        ppr::gpu::SOpenCLConfig& opencl,
        SDataStat& stat,
        tbb::task_arena& arena,
        std::vector<int>& histogram,
        void (*process_chunk) (SHistogram& hist, SConfig&, ppr::gpu::SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned long long, double*, std::vector<int>&))
    {
        DWORD granulatity = m_allocationGranularity * m_scale;

        // Create a file
        const WCHAR* filename = char2wchar(m_filename);
        HANDLE hfile = ::CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ,
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
                        unsigned long long data_in_chunk = granulatity / sizeof(double);

                        // Set opencl computing limits
                        if (opencl.wg_size != 0)
                        {
                            // Get number of data, which we want to process on GPU
                            opencl.wg_count = static_cast<unsigned long>(data_in_chunk / opencl.wg_size);
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
        delete filename;
    }


}