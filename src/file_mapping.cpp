#include "file_mapping.h"
#define BUFFER_SIZE 100 // TODO: change

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

        m_fileLen = GetFileSize(m_file, 0);
        m_size = m_fileLen / sizeof(double);

        CloseHandle(m_file);

        /*bool res_mf = MapFile();
        if (!res_mf)
        {
            return;
        }
        view();*/
    }

    bool FileMapping::CreateFile_n()
    {
        size_t i;
        char* path = (char*)malloc(BUFFER_SIZE);

        // Conversion
        wcstombs_s(&i, path, (size_t)BUFFER_SIZE,
            m_filename, (size_t)BUFFER_SIZE - 1); // -1 so the appended NULL doesn't fall outside the allocated buffer

        if (path)
        {
            m_file = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
            // Free multibyte character buffer 
            free(path);
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

    // Calls ProcessChunk with each chunk of the file.
    // https://stackoverflow.com/questions/9889557/mapping-large-files-using-mapviewoffile
    void FileMapping::ReadInChunks(
        SConfig& config,
        SOpenCLConfig& opencl,
        SDataStat& stat,
        tbb::task_arena& arena,
        std::vector<double>& histogram,
        void (*ProcessChunk) (SConfig&, SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned int, double*, std::vector<double>&))
    {
        // Offsets must be a multiple of the system's allocation granularity.  We
        // guarantee this by making our view size equal to the allocation granularity.
        SYSTEM_INFO sysinfo = { 0 };
        ::GetSystemInfo(&sysinfo);
        double scale = MAX_FILE_SIZE_MEM / sysinfo.dwAllocationGranularity;
        m_allocationGranularity = sysinfo.dwAllocationGranularity * scale;


        HANDLE hfile = ::CreateFileW(m_filename, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hfile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER file_size = { 0 };
            ::GetFileSizeEx(hfile, &file_size);
            const unsigned long long cbFile =
                static_cast<unsigned long long>(file_size.QuadPart);

            HANDLE hmap = ::CreateFileMappingW(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
            if (hmap != NULL) {
                for (unsigned long long offset = 0; offset < cbFile; offset += m_allocationGranularity) {
                    DWORD high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
                    DWORD low = static_cast<DWORD>(offset & 0xFFFFFFFFul);
                    // The last view may be shorter.
                    if (offset + m_allocationGranularity > cbFile) {
                        m_allocationGranularity = static_cast<int>(cbFile - offset);
                    }

                    double* pView = static_cast<double*>(
                        ::MapViewOfFile(hmap, FILE_MAP_READ, high, low, m_allocationGranularity));

                    if (pView != NULL) {
                        //ProcessChunk(pView, cbView);
                        unsigned int data_in_chunk = m_allocationGranularity / sizeof(double);

                        // Get number of data, which we want to process on GPU
                        opencl.wg_count = data_in_chunk / opencl.wg_size;
                        opencl.data_count_for_gpu = data_in_chunk - (data_in_chunk % opencl.wg_size);

                        // The rest of the data we will process on CPU
                        opencl.data_count_for_cpu = opencl.data_count_for_gpu + 1;

                        // Run
                        ProcessChunk(config, opencl, stat, arena, data_in_chunk, pView, histogram);

                        UnmapViewOfFile(pView);
                    }
                }
                ::CloseHandle(hmap);
            }
            ::CloseHandle(hfile);
        }
    }



}