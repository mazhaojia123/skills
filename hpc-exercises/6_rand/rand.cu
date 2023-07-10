#include <stdio.h>
#include <stdlib.h>

static bool haveSrand = false;

void simpleRand() {
	for (int i = 0; i < 10; i++)
		printf("%d  ", rand());
	printf("\n");
}

void getRand(int low, int upper) {
	for (int i = 0; i < 100; i++)
		printf("%d ", rand() % (upper - low + 1) + low); // 闭区间
	printf("\n");
	for (int i = 0; i < 100; i++)
		printf("%d ", rand() % (upper - low) + low); // 开区间
	printf("\n");
}

void getSignedRand(int upper) {
	int low = 0;
	for (int i = 0; i < 20; i++) {
		int tmp = rand() % (upper - low) + low;
		tmp = tmp * (rand() % 2 == 0 ? 1 : -1);
		printf("%d ", tmp);
	}
	printf("\n");
}

void getRandFloat(int low, int upper) {
	// 主要的思路是将这个数给放缩到某个范围内
	for (int i = 0; i < 20; i++) {
		float tmp = (float)rand() / RAND_MAX * (upper - low) + low;
		printf("%.2f ", tmp);
	}
	printf("\n");

}

int main() {
	if (!haveSrand)
		srand(time(NULL)); 		// NOTE: 这个句子只能调用一次，否则会导致出现重复的数字
	// simpleRand();
	// simpleRand();
	// simpleRand();

	// getRand(30, 100);

	// getSignedRand(5);
	// getSignedRand(5);
	// getSignedRand(5);

	getRandFloat(10, 100);
	getRandFloat(10, 100);
	getRandFloat(10, 100);

	return 0;
}