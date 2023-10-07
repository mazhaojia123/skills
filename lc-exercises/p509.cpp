#include <iostream>

class Solution {
public:
    int fib(int n) {
		if (n == 0) return 0;
		if (n == 1) return 1;

		int dp1 = 0;
		int dp2 = 1;
		int res = 0;

		for (int i = 2; i <= n; i++) {
			res = dp1 + dp2; // 递推关系
			dp1 = dp2;	
			dp2 = res;
		}

		return res;
    }
};

int main() {
	Solution s;
	for (int i = 0; i <= 10; i++) {
		std::cout << s.fib(i) << '\t';
	}

}