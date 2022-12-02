#include "file_mapping.h"
#include "smp/smp_utils.h"

#include<future>

namespace ppr
{

    FileMapping::FileMapping(const WCHAR* filename)
        : m_filename(filename), m_file(INVALID_HANDLE_VALUE), m_mapping(INVALID_HANDLE_VALUE), m_data(NULL)
    {
        bool res_cf = CreateFile_n();
        if (!res_cf)
        {
            return;
        }
        LARGE_INTEGER file_size = { 0 };
        ::GetFileSizeEx(m_file, &file_size);
        m_fileLen = static_cast<unsigned long long>(file_size.QuadPart);
        m_size = m_fileLen / sizeof(double);

        bool res_mf = MapFile();
        if (!res_mf)
        {
            return;
        }
        view();
    }

    FileMapping::FileMapping(SConfig& config)
        : m_filename(config.input_fn), m_file(INVALID_HANDLE_VALUE), m_mapping(INVALID_HANDLE_VALUE), m_data(NULL)
    {
        bool res_cf = CreateFile_n();
        if (!res_cf)
        {
            return;
        }

        // Offsets must be a multiple of the system's allocation granularity.  We
        // guarantee this by making our view size equal to the allocation granularity.
        SYSTEM_INFO sysinfo = { 0 };
        ::GetSystemInfo(&sysinfo);
        double scale = static_cast<double>(MAX_FILE_SIZE_MEM) / sysinfo.dwAllocationGranularity;
        m_allocationGranularity = sysinfo.dwAllocationGranularity * scale;

        LARGE_INTEGER file_size = { 0 };
        ::GetFileSizeEx(m_file, &file_size);
        m_fileLen = static_cast<unsigned long long>(file_size.QuadPart);
        m_size = m_fileLen / sizeof(double);

        CloseHandle(m_file);
    }

    bool FileMapping::CreateFile_n()
    {
        size_t i;
        char* path = (char*)malloc(100);

        // Conversion
        wcstombs_s(&i, path, (size_t)100,
            m_filename, (size_t)100 - 1); // -1 so the appended NULL doesn't fall outside the allocated buffer

        if (path)
        {
            m_file = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
            // Free multibyte character buffer 
            delete[] path;
        }
        if (m_file == INVALID_HANDLE_VALUE)
        {
            return false;
        };

        return true;
    }

    bool FileMapping::MapFile() {

        m_mapping = CreateFileMapping(m_file, 0, PAGE_READONLY, 0, 0, 0);
        if (m_mapping == 0)
        {
            CloseHandle(m_file);
            return false;
        };

        return true;
    }

    void FileMapping::view()
    {
        m_data = (double*)MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
    }

    double* FileMapping::GetData() const
    {
        return m_data;
    }

    const unsigned int FileMapping::GetFileLen() const
    {
        return m_fileLen;
    }

    const unsigned int FileMapping::GetCount() const
    {
        return m_size;
    }

    void FileMapping::UnmapFile()
    {
        UnmapViewOfFile(m_data);
        CloseHandle(m_mapping);
        CloseHandle(m_file);
    }

    const DWORD FileMapping::GetGranularity() const
    {
        return m_allocationGranularity;
    }

    void FileMapping::ReadInChunksHist(
        SHistogram& hist,
        SConfig& config,
        SOpenCLConfig& opencl,
        SDataStat& stat,
        std::vector<int>& histogram)
    {

        DWORD granulatity = m_allocationGranularity;

        HANDLE hfile = ::CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hfile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER file_size = { 0 };
            ::GetFileSizeEx(hfile, &file_size);
            const unsigned long long cbFile =
                static_cast<unsigned long long>(file_size.QuadPart);

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

                    // Create a chunk
                    double* pView = static_cast<double*>(
                        ::MapViewOfFile(hmap, FILE_MAP_READ, high, low, granulatity));

                    if (pView != NULL) {
                        unsigned int data_in_chunk = granulatity / sizeof(double);

                        // Compute data count
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
                            opencl.data_count_for_cpu = data_in_chunk / config.thread_count;
                        }

                        std::vector<std::future<std::tuple<std::vector<int>, double>>> workers(config.thread_count);

                        // Process chunk with multuply threads
                        for (int i = 0; i < config.thread_count; i++)
                        {
                            ppr::parallel::CHistProcessingUnit unit(hist, config, opencl, stat);
                            workers[i] = std::async(std::launch::async | std::launch::deferred, &ppr::parallel::CHistProcessingUnit::RunCPU, unit, pView + (opencl.data_count_for_cpu * i), opencl.data_count_for_cpu);
                        }

                        // Agregate results results
                        for (auto& worker : workers)
                        {
                            auto [vector, variance] = worker.get();
                            stat.variance += variance;
                            std::transform(histogram.begin(), histogram.end(), vector.begin(), histogram.begin(), std::plus<int>());
                        }

                        UnmapViewOfFile(pView);
                    }
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
    }

    // Calls ProcessChunk with each chunk of the file.
    // https://stackoverflow.com/questions/9889557/mapping-large-files-using-mapviewoffile
    void FileMapping::ReadInChunksStat(
        SConfig& config,
        SOpenCLConfig& opencl,
        SDataStat& stat)
    {
        DWORD granulatity = m_allocationGranularity;
                                
        HANDLE hfile = ::CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hfile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER file_size = { 0 };
            ::GetFileSizeEx(hfile, &file_size);
            const unsigned long long cbFile =
                static_cast<unsigned long long>(file_size.QuadPart);

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

                    // Create a chunk
                    double* pView = static_cast<double*>(
                        ::MapViewOfFile(hmap, FILE_MAP_READ, high, low, granulatity));

                    if (pView != NULL) {
                        unsigned int data_in_chunk = granulatity / sizeof(double);

                        // Compute data count
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
                            opencl.data_count_for_cpu = data_in_chunk / config.thread_count;
                        }
                        std::vector<std::future<SDataStat>> workers(config.thread_count);
                        // Process chunk with multuply threads

                        if (config.mode == ERun_mode::SMP)
                        {
                            for (int i = 0; i < config.thread_count; i++)
                            {
                                ppr::parallel::CStatProcessingUnit unit(config, opencl);
                                workers[i] = std::async(std::launch::async | std::launch::deferred, &ppr::parallel::CStatProcessingUnit::RunCPU, unit, pView + (opencl.data_count_for_cpu * i), opencl.data_count_for_cpu);
                            }
                        }
                        else
                        {
                            unsigned long count = opencl.data_count_for_gpu / config.thread_count;
                            for (int i = 0; i < config.thread_count; i++)
                            {
                                ppr::parallel::CStatProcessingUnit unit(config, opencl);
                                workers[i] = std::async(std::launch::async | std::launch::deferred, &ppr::parallel::CStatProcessingUnit::RunGPU, unit, pView + (count * i), count);
                            }
                        }

                        // Agregate results results
                        for (auto& worker : workers)
                        {
                            SDataStat local_stat = worker.get();
                            stat.sum += local_stat.sum;
                            stat.n += local_stat.n;
                            stat.min = std::min({ stat.min, std::min({stat.min, local_stat.min}) });
                            stat.max = std::max({ stat.max, std::max({ stat.max, local_stat.max }) });
                        }
                       

                        UnmapViewOfFile(pView);
                    }
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
    }


    // Calls ProcessChunk with each chunk of the file.
    // https://stackoverflow.com/questions/9889557/mapping-large-files-using-mapviewoffile
    void FileMapping::ReadInChunks(
        SHistogram& hist,
        SConfig& config,
        SOpenCLConfig& opencl,
        SDataStat& stat,
        tbb::task_arena& arena,
        std::vector<int>& histogram,
        void (*ProcessChunk) (SHistogram& hist, SConfig&, SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned int, double*, std::vector<int>&))
    {
        DWORD granulatity = m_allocationGranularity;

        HANDLE hfile = ::CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hfile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER file_size = { 0 };
            ::GetFileSizeEx(hfile, &file_size);
            const unsigned long long cbFile =
                static_cast<unsigned long long>(file_size.QuadPart);

            HANDLE hmap = ::CreateFileMappingW(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (hmap != NULL) {
                for (unsigned long long offset = 0; offset < cbFile; offset += granulatity) {
                    DWORD high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
                    DWORD low = static_cast<DWORD>(offset & 0xFFFFFFFFul);
                    // The last view may be shorter.
                    if (offset + granulatity > cbFile) {
                        granulatity = static_cast<int>(cbFile - offset);
                    }

                    double* pView = static_cast<double*>(
                        ::MapViewOfFile(hmap, FILE_MAP_READ, high, low, granulatity));

                    if (pView != NULL) {
                        //ProcessChunk(pView, cbView);
                        unsigned int data_in_chunk = granulatity / sizeof(double);

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
                        ProcessChunk(hist, config, opencl, stat, arena, data_in_chunk, pView, histogram);

                        UnmapViewOfFile(pView);
                    }
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
    }



}