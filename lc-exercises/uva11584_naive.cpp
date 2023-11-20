#include <iostream>
#include <string>

using namespace std;

int f[1001][1001];

// 判断s[i:j]是不是回文的
bool is_palindromic(const string &s, int i, int j){
	while (i<=j) {
		if (s[i] != s[j]) return false;
		i++,j--;
	}
	return true;
}

int main() {
	freopen("../uva11584.in", "r", stdin);
	int case_n;
	string input;

	cin >> case_n;
	for (int case_cnt = 0; case_cnt < case_n; case_cnt++) {
		cin >> input;
		int n = input.size();
		for (int l = 1; l <= n; l++) {
			for (int i = 1; i <= n - l + 1; i++) {
				int j = i + l - 1;
				f[i][j] = j-i+1;
				if (is_palindromic(input,i-1,j-1)) {
					f[i][j] = 1;
				} else {
					for (int k = i; k <= j - 1; k++) {
						f[i][j] = min(f[i][j], f[i][k] + f[k+1][j]);
					}
				}
			}
		}	
		// for (int i = 1; i <= input.size(); i++) {
		// 	for (int j = 1; j <= input.size(); j++) {
		// 		printf("%d\t", f[i][j]);
		// 	}
		// 	printf("\n");
		// }
		cout << f[1][input.size()] << endl;
	}

	return 0;
}

/*

该做法的时间复杂度分析：
case_cnt * n * n * n
n是1000
1e9 还有常数项，所以G了

*/