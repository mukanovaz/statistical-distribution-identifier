#pragma once
#ifndef FILEMAP_H
#define FILEMAP_H

#include "smp/smp_utils.h"
#include "config.h"
#include "data.h"

#include<future>
#include<vector>
#include <windows.h>
#include <iostream>
#undef min
#undef max

#include <tbb/task_arena.h>
#include <tbb/tick_count.h>

namespace ppr
{
    class File_mapping
    {
        private:
            /// <summary>
            /// Handle to file
            /// </summary>
            HANDLE m_file;
            /// <summary>
            /// Handle for file mapping
            /// </summary>
            HANDLE m_mapping;
            /// <summary>
            /// Scale is using to multiply alocation granularity
            /// </summary>
            double m_scale;
            /// <summary>
            /// Maped data
            /// </summary>
            double* m_data;
            /// <summary>
            /// File name
            /// </summary>
            const WCHAR* m_filename;
            /// <summary>
            /// File lenght
            /// </summary>
            unsigned long long m_fileLen;
            /// <summary>
            /// Data count in a file
            /// </summary>
            DWORD m_size;
            /// <summary>
            /// System allocation granularity
            /// </summary>
            DWORD m_allocationGranularity;

            /// <summary>
            /// Create file. Is using for getting file lenght before all computings and for sequential computing.
            /// </summary>
            /// <returns>Is success</returns>
            bool create_file_n();

            /// <summary>
            /// Map file into a memory. (Using only for sequential computing)
            /// </summary>
            /// <returns>Is success</returns>
            bool map_file();

            /// <summary>
            /// Create a file view. (Using only for sequential computing)
            /// </summary>
            void view();

        public:
            /// <summary>
            /// Constructor is using for sequential computing
            /// </summary>
            /// <param name="filename">file name</param>
            File_mapping(const WCHAR* filename);

            /// <summary>
            /// Main constructor
            /// </summary>
            /// <param name="config">program configuration structure</param>
            File_mapping(SConfig& config);

            /// <summary>
            /// Get data from mapped file. (Using only for sequential computing)
            /// </summary>
            /// <returns>64-bit double array pointer</returns>
            double* get_data() const;

            /// <summary>
            /// Unmap file. (Using only for sequential computing)
            /// </summary>
            void unmap_file();

            /// <summary>
            /// Get allocation granularity for current system
            /// </summary>
            /// <returns>allocation granularity</returns>
            const DWORD get_granularity() const;

            /// <summary>
            /// Get file lenght in bytes
            /// </summary>
            /// <returns>file lenght in bytes</returns>
            const unsigned int get_file_len() const;

            /// <summary>
            /// Returnes how many doubles exitst in input file
            /// </summary>
            /// <returns>Number of doubles</returns>
            const unsigned int get_count() const;

            /// <summary>
            /// Calls 'process_chunk' function with each chunk of the file.
            /// Mapping data bloks (chunks) and make computings using function pointer in a input parametr for each chunk of the file.
            /// (Not optimized solution)
            /// https://stackoverflow.com/questions/9889557/mapping-large-files-using-mapviewoffile
            /// </summary>
            /// <param name="hist">histogram configuration structure</param>
            /// <param name="config">program configuration structure</param>
            /// <param name="opencl">opencl configuration structure</param>
            /// <param name="stat">statistics structure</param>
            /// <param name="arena">arena object. using for TBB algorithm</param>
            /// <param name="histogram">vector reference for frequency histogram</param>
            /// <param name="process_chunk">method is using for process one data chunk</param>
            void read_in_chunks(
                SHistogram& hist,
                SConfig& config,
                SOpenCLConfig& opencl, 
                SDataStat& stat, 
                tbb::task_arena& arena, 
                std::vector<int>& histogram,
                void (*process_chunk) (SHistogram& hist, SConfig&, SOpenCLConfig&, SDataStat&, tbb::task_arena&, unsigned int, double*, std::vector<int>&));

            /// <summary>
            /// Mapping data bloks (chunks) and collecting statistics of these data using multiply threads
            /// </summary>
            /// <param name="config">program configuration structure</param>
            /// <param name="opencl">opencl configuration structure</param>
            /// <param name="stat">statistics structure</param>
            void read_in_chunks_stat(
                SConfig& config,
                SOpenCLConfig& opencl,
                SDataStat& stat);

            /// <summary>
            /// Mapping data bloks (chunks) and creating frequency histogram of these data using multiply threads
            /// </summary>
            /// <param name="hist">histogram configuration structure</param>
            /// <param name="config">program configuration structure</param>
            /// <param name="opencl">opencl configuration structure</param>
            /// <param name="stat">statistics structure</param>
            /// <param name="histogram">vector reference for frequency histogram</param>
            void read_in_chunks_hist(
                SHistogram& hist,
                SConfig& config,
                SOpenCLConfig& opencl,
                SDataStat& stat,
                std::vector<int>& histogram);

    };
}
#endif