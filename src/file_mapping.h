#pragma once
#ifndef FILEMAP_H
#define FILEMAP_H
#include <windows.h>
#include <iostream>

class FileMapping
{
    private:
        HANDLE File;
        HANDLE Mapping;
        const double* Data;
        const char* Filename;
        unsigned int FileLen;
        unsigned int DoublesCount;

        bool CreateFile_n();

        bool MapFile();

        void view();

    public:
        FileMapping(const char* filename);

        const double* GetData() const;

        const unsigned int GetFileLen() const;

        const unsigned int GetCount() const;

        void UnmapFile();
};
#endif