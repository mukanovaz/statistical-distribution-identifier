#include "file_mapping.h"

FileMapping::FileMapping(const char* filename) 
    : Filename(filename), File(INVALID_HANDLE_VALUE), Mapping(INVALID_HANDLE_VALUE), Data(NULL)
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
    File = CreateFile(Filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (File == INVALID_HANDLE_VALUE)
    {
        return false;
    };

    return true;
}

bool FileMapping::MapFile() {

    Mapping = CreateFileMapping(File, 0, PAGE_READONLY, 0, 0, 0);
    if (Mapping == 0)
    {
        CloseHandle(File);
        return false;
    };

    return true;
}

void FileMapping::view()
{
    FileLen = GetFileSize(File, 0);
    DoublesCount = FileLen / sizeof(double);

    Data = (const double*)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
}

const double* FileMapping::GetData() const
{
    return Data;
}

const unsigned int FileMapping::GetFileLen() const
{
    return FileLen;
}

const unsigned int FileMapping::GetCount() const
{
    return DoublesCount;
}

void FileMapping::UnmapFile()
{
    UnmapViewOfFile(Data);
    CloseHandle(Mapping);
    CloseHandle(File);
}


