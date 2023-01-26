#include "include/file_mapper.h"

namespace ppr
{
    File_mapper* File_mapper::instance = NULL;

    const unsigned long long File_mapper::get_file_len()
    {
        return m_fileLen;
    }

    double* File_mapper::view(DWORD high, DWORD low, DWORD granulatity)
    {
        return (double*)MapViewOfFile(m_mapping, FILE_MAP_READ, high, low, granulatity);
    }

    void File_mapper::init(const char* filename)
	{
        m_filename = filename;
        m_file = INVALID_HANDLE_VALUE;
        m_mapping = INVALID_HANDLE_VALUE;

        bool res_cf = create_file_n();
        if (!res_cf)
        {
            return;
        }

        // Get file size and number of doubles inside
        LARGE_INTEGER file_size = { 0 };
        ::GetFileSizeEx(m_file, &file_size);
        m_fileLen = static_cast<unsigned long long>(file_size.QuadPart);
        m_size = static_cast<DWORD>(m_fileLen / sizeof(double));

        // Create file mapping
        bool res_mf = map_file();
        if (!res_mf)
        {
            ppr::print_error("Cannot create a mapping");
            CloseHandle(m_file);
            return;
        }

        // Offsets must be a multiple of the system's allocation granularity.  We
        // guarantee this by making our view size equal to the allocation granularity.
        SYSTEM_INFO sysinfo = { 0 };
        ::GetSystemInfo(&sysinfo);

        m_allocationGranularity = sysinfo.dwAllocationGranularity;
	}

    void File_mapper::close_all()
    {
        CloseHandle(m_mapping);
        CloseHandle(m_file);
    }

    bool File_mapper::create_file_n()
    {
        const WCHAR* filename = char2wchar(m_filename);
        m_file = CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

        if (m_file == INVALID_HANDLE_VALUE)
        {
            delete filename;
            return false;
        };

        return true;
    }

    bool File_mapper::map_file() {

        m_mapping = CreateFileMappingW(m_file, 0, PAGE_READONLY, 0, 0, 0);
        if (m_mapping == 0)
        {
            CloseHandle(m_file);
            return false;
        };

        return true;
    }

    const long File_mapper::get_granularity()
    {
        return m_allocationGranularity;
    }

	File_mapper* ppr::File_mapper::get_instance()
	{
		if (instance == NULL)
		{
			instance = new File_mapper();
		}
		
		return (instance);
	}
}

