#pragma once
#ifndef FILEMAP_H
#define FILEMAP_H
#include <windows.h>
#include <iostream>
#undef min
#undef max

#include <tbb/tick_count.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/blocked_range.h>

#include "config.h"
#include "data.h"


namespace ppr
{
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

            void ReadInChunks(
                SConfig& config,
                SOpenCLConfig& opencl, 
                SDataStat& stat, 
                tbb::task_arena& arena, 
                std::vector<double>& histogram,
                void (*ProcessChunk) (SConfig&, SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned int, double*, std::vector<double>&));

    };
}
#endif