#include <windows.h>
#include <iostream>
#include "file_mapping.h"
#include <vector>

class FileMapping
{
    private:
        HANDLE File;
        HANDLE Mapping;
        const double* Data;
        const char* Filename;
        unsigned int FileLen;
        unsigned int DoublesCount;

        bool create_file()
        {
            File = CreateFile(Filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
            if (File == INVALID_HANDLE_VALUE)
            {
                return false;
            };

            return true;
        }

        bool map_file() {

            Mapping = CreateFileMapping(File, 0, PAGE_READONLY, 0, 0, 0);
            if (Mapping == 0)
            {
                CloseHandle(File);
                return false;
            };

            return true;
        }

        void view()
        {
            FileLen = GetFileSize(File, 0);
            DoublesCount = FileLen / sizeof(double);

            Data = (const double*)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
        }

    public:
        FileMapping(const char* filename) : Filename(filename), File(INVALID_HANDLE_VALUE), Mapping(INVALID_HANDLE_VALUE), Data(NULL)
        {
            bool res_cf = create_file();
            if (!res_cf)
            {
                return;
            }
            bool res_mf = map_file();
            if (!res_mf)
            {
                return;
            }
            view();
        }

        const double* get_data() const  
        {
            return Data;
        }

        const unsigned int get_filelen() const
        {
            return FileLen;
        }

        const unsigned int get_count() const
        {
            return DoublesCount;
        }

        void unmap_file()
        {
            UnmapViewOfFile(Data);
            CloseHandle(Mapping);
            CloseHandle(File);
        }
};



