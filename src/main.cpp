#include "file_mapping.h"

int main(int argc, char** argv) {

	//std::wcout << L"vystup programu" << std::endl;
	
	std::unique_ptr<HANDLE> file = create_file(argv[1]);
	std::unique_ptr<HANDLE> mapping = map_file(*file);
	const double* data = get_data(*file, *mapping);
	unmap_file(data, *file, *mapping);

	mapping.reset();
	file.reset();

	int ret = getchar();
	return ret;
}