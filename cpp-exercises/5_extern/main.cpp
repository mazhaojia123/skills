#include "hello.hpp"
#include <iostream>

extern int gpu_index;

int main() {
	std::cout << "gpu_index: " << gpu_index << std::endl;
}
