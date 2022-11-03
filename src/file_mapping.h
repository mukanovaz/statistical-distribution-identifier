#pragma once
#include <windows.h>
#include <iostream>


const std::unique_ptr <HANDLE> create_file(const char* filename);

const std::unique_ptr <HANDLE> map_file(const HANDLE& file);

const double* get_data(const HANDLE& file, const HANDLE& mapping);

void unmap_file(const double* data, const HANDLE& file, const HANDLE& mapping);