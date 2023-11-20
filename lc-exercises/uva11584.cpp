#include <iostream>
#include <string>

using namespace std;

int dp[1001];

// 判断s[i:j]是不是回文的
bool is_palindromic(const string &s, int i, int j){
	while (i<=j) {
		if (s[i] != s[j]) return false;
		i++,j--;
	}
	return true;
}

int main() {
	// freopen("../uva11584.in", "r", stdin);
	int case_n;
	string input;

	cin >> case_n;
	// NOTE: 1. 遍历长度；2. 遍历起点终点
	for (int case_cnt = 0; case_cnt < case_n; case_cnt++) {
		cin >> input;
		int len = input.size();

		// NOTE: 3. 遍历区间中间的每个点
		for (int i = 0; i <= len-1; i++) {
			dp[i] = i+1;
			for (int j = 0; j <= i; j++) {
				if (is_palindromic(input, j, i)) {
					dp[i] = min(dp[i], dp[j-1] + 1);
				}
			}
		}
		cout << dp[len-1] << endl;
	}

	return 0;
}