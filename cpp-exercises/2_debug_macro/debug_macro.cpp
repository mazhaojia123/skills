#include <iostream>

#define INTERCEPT(devPtr,msgStr,numOfVar)		\
{                                          		\
	printf("%d, %s, %d\n", (devPtr), (msgStr), (numOfVar)); \
} 													\

int main() {
	INTERCEPT(12, "hello", 15);
}