#include "file_mapping.h"

FileMapping::FileMapping(const char* filename) 
    : m_filename(filename), m_file(INVALID_HANDLE_VALUE), m_mapping(INVALID_HANDLE_VALUE), m_data(NULL)
{
    bool res_cf = CreateFile_n();
    if (!res_cf)
    {
        return;
    }
    bool res_mf = MapFile();
    if (!res_mf)
    {
        return;
    }
    view();
}

bool FileMapping::CreateFile_n()
{
    m_file = CreateFile(m_filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
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
    m_fileLen = GetFileSize(m_file, 0);
    m_size = m_fileLen / sizeof(double);

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
void FileMapping::ReadInChunks(const WCHAR* pszFileName) {
    // Offsets must be a multiple of the system's allocation granularity.  We
    // guarantee this by making our view size equal to the allocation granularity.
    SYSTEM_INFO sysinfo = { 0 };
    ::GetSystemInfo(&sysinfo);
    DWORD cbView = sysinfo.dwAllocationGranularity;

    HANDLE hfile = ::CreateFileW(pszFileName, GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, 0, NULL);
    if (hfile != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER file_size = { 0 };
        ::GetFileSizeEx(hfile, &file_size);
        const unsigned long long cbFile =
            static_cast<unsigned long long>(file_size.QuadPart);

        HANDLE hmap = ::CreateFileMappingW(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hmap != NULL) {
            for (unsigned long long offset = 0; offset < cbFile; offset += cbView) {
                DWORD high = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFul);
                DWORD low = static_cast<DWORD>(offset & 0xFFFFFFFFul);
                // The last view may be shorter.
                if (offset + cbView > cbFile) {
                    cbView = static_cast<int>(cbFile - offset);
                }



                const double* pView = static_cast<const double*>(
                    ::MapViewOfFile(hmap, FILE_MAP_READ, high, low, cbView));

                MEMORY_BASIC_INFORMATION mbi = { 0 };
                VirtualQueryEx(GetCurrentProcess(), pView, &mbi, sizeof(mbi));

                if (pView != NULL) {
                    //ProcessChunk(pView, cbView);
                    double test = 0;
                }
            }
            ::CloseHandle(hmap);
        }
        ::CloseHandle(hfile);
    }
}


