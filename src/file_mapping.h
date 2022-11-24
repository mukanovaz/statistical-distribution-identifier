#pragma once
#ifndef FILEMAP_H
#define FILEMAP_H
#include <windows.h>
#include <iostream>

class FileMapping
{
    private:
        HANDLE m_file;
        HANDLE m_mapping;
        double* m_data;
        const char* m_filename;
        unsigned int m_fileLen;
        unsigned int m_size;

        bool CreateFile_n();

        bool MapFile();

        void view();

    public:
        FileMapping(const char* filename);

        double* GetData() const;

        const unsigned int GetFileLen() const;

        const unsigned int GetCount() const;

        void UnmapFile();
};
#endif