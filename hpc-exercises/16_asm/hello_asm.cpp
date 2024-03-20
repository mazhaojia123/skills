#include <iostream>
using namespace std;

int main() {
    int res;
    // move value to register %eax
    // move value to register %ebx
    // subtracting and storing result in res

	asm volatile("# asm begin");
    __asm__(
        "movl $20, %%eax;"
        "movl $10, %%ebx;"
        "subl %%ebx, %%eax "
        : "=a"(res));
    cout << "res " << res << endl;
	asm volatile("# asm end");
    return 0;
}