#include <stdio.h>

#include "gpu_index.hpp"
#include "cpu_index.h"

void a_gpu_index() {
	printf("a(C)'s gpu_index = %d\n", gpu_index);
	printf("a(C)'s cpu_index = %d\n", cpu_index);
}