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

    m_data = (const double*)MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
}

const double* FileMapping::GetData() const
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


