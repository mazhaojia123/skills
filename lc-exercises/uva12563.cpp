#include <iostream>
#include <cstring>
#include <cstdio>
// #define DEBUG

int T, n, target;
int dp[51][10000];
int num[51];

int main() {
#ifdef DEBUG
	freopen("../A.in", "r", stdin);
#endif
	std::cin >> T;
	for (int test_cnt = 1; test_cnt <= T; test_cnt++) {
		std::cin >> n >> target;
		for (int i = 1; i <= n; i++) {
			std::cin >> num[i];
		}	
		
		memset(dp, 0xc0, sizeof(dp));
		for (int i = 0; i <= n; i++) dp[i][0] = 0;
		for (int i = 1; i <= n; i++) {
			for (int j = 0; j <= target - 1; j++) {
				dp[i][j] = dp[i-1][j];
				if (num[i] <= j) {
					dp[i][j] = std::max(dp[i][j], dp[i-1][j - num[i]]+1);
				}
			}
		}

		int max_res = -1, max_index = -1;
		for (int j = target-1; j >= 0; j--) {
			// printf("%d\n", dp[n][j]);
			if (max_res < dp[n][j]) {
				max_res = dp[n][j];
				max_index = j;
			}
		}

		printf("Case %d: %d %d\n", test_cnt, max_res+1, max_index+678);
	}

	return 0;
}