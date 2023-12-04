// 只能想到动态规划的解法
#include <iostream>
using namespace std;

int nums[101][101];
int f[101][101];
int N;

int main() {	
	// freopen("../uva108.in", "r", stdin);
	cin >> N;
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			cin >> nums[i][j];
			f[i][j] = f[i-1][j] + nums[i][j];
		}
	}

	int res = -128;

	for (int i = 1; i <= N; i++) {
		for (int j = i; j <= N; j++) {
			int tmp = 0;
			for (int k = 1; k <= N; k++) {
				tmp += f[j][k] - f[i-1][k];
				if (tmp <= 0) tmp = 0;
				else {
					if (tmp > res) res = tmp;
				}
			}
		}
	}
	printf("%d\n", res);
}