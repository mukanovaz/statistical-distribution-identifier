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


#include<future>
#include<vector>


namespace ppr
{
    class FileMapping
    {
        private:
            HANDLE m_file;
            HANDLE m_mapping;
            double* m_data;
            const WCHAR* m_filename;
            unsigned int m_fileLen;
            unsigned int m_size;
            DWORD m_allocationGranularity;

            bool CreateFile_n();

            bool MapFile();

            void view();

        public:
            FileMapping(const WCHAR* filename);

            double* GetData() const;

            const DWORD GetGranularity() const;

            const unsigned int GetFileLen() const;

            const unsigned int GetCount() const;

            void UnmapFile();

            void ReadInChunks(
                SHistogram& hist,
                SConfig& config,
                SOpenCLConfig& opencl, 
                SDataStat& stat, 
                tbb::task_arena& arena, 
                std::vector<int>& histogram,
                void (*ProcessChunk) (SHistogram& hist, SConfig&, SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned int, double*, std::vector<int>&));

            void ReadInChunksStat(
                SConfig& config,
                SOpenCLConfig& opencl,
                SDataStat& stat);

            void ReadInChunksHist(
                SHistogram& hist,
                SConfig& config,
                SOpenCLConfig& opencl,
                SDataStat& stat,
                std::vector<int>& histogram);

    };
}
#endif