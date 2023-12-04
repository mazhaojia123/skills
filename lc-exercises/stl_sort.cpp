#include <iostream>
#include <algorithm>

int main() {
	int a[] = {7,1,4,1,3,10};
	printf("sizeof_a: %lld\n", sizeof(a)/sizeof(int));
	int len = sizeof(a)/sizeof(int);
	// std::sort(a, a+len, std::greater<int>());
	std::sort(a, a+len, std::less<int>());
	for (int i = 0; i < len; i++) {
		printf("%d ", a[i]);
	}
	puts("");
}