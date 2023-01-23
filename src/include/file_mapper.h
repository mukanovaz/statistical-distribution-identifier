#include "config.h"
#include "data.h"

namespace ppr
{
    class File_mapper
    {
        private:
            static File_mapper* instance;
            File_mapper() {};

        protected:
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
            long m_scale;
            /// <summary>
            /// Maped data
            /// </summary>
            double* m_data;
            /// <summary>
            /// File name
            /// </summary>
            const char* m_filename;
            /// <summary>
            /// File lenght
            /// </summary>
            unsigned long long m_fileLen;
            /// <summary>
            /// Data count in a file
            /// </summary>
            long m_size;
            /// <summary>
            /// System allocation granularity
            /// </summary>
            long m_allocationGranularity;

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


        public:
            /// <summary>
            /// Get allocation granularity for current system
            /// </summary>
            /// <returns>allocation granularity</returns>
            const long get_granularity();

            /// <summary>
            /// Get file lenght in bytes
            /// </summary>
            /// <returns>file lenght in bytes</returns>
            const unsigned long long get_file_len();

            /// <summary>
            /// Create a file view. (Using only for sequential computing)
            /// </summary>
            double* view(DWORD high, DWORD low, DWORD granulatity);
            void init(const char* filename);

            static File_mapper* get_instance();
    
    };
}