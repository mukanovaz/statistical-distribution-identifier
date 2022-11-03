#include <windows.h>
#include <iostream>
#include "file_mapping.h"
#include <vector>


const std::unique_ptr <HANDLE> create_file(const char* filename)
{
    HANDLE file = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (file == INVALID_HANDLE_VALUE)
    {
        return NULL;
    };

    return std::make_unique<HANDLE>(file);
}

const std::unique_ptr<HANDLE> map_file(const HANDLE& file) {

    HANDLE mapping = CreateFileMapping(file, 0, PAGE_READONLY, 0, 0, 0);
    if (mapping == 0) 
    { 
        CloseHandle(file);
        return NULL; 
    }

    return std::make_unique<HANDLE>(mapping);
}

const double* get_data(const HANDLE& file, const HANDLE& mapping)
{
    unsigned int len = GetFileSize(file, 0);
    unsigned int count = len / sizeof(double);

    const double* data = (const double*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (data)
    {
        //// need volatile or need to use result - compiler will otherwise optimize out whole loop
        //volatile unsigned int touch = 0;

        //for (unsigned int i = 0; i < count; i++)
        //{
        //    double d = (double)data[i];
        //    std::cout << d << std::endl;
        //}
    }
    return data;
}

void unmap_file(const double* data, const HANDLE& file, const HANDLE& mapping)
{
    UnmapViewOfFile(data);
    CloseHandle(mapping);
    CloseHandle(file);
}

