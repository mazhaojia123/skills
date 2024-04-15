extern "C" {
#include "b.hpp"
}
#include "gpu_index.hpp"
#include "cpu_index.h"
#include <iostream>

void b_gpu_index() {
	std::cout << "b(C++)'s gpu_index = " << gpu_index << std::endl;
	std::cout << "b(C++)'s cpu_index = " << cpu_index << std::endl;
}