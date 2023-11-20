#include <iostream>
#include <vector>
#include <climits>

using namespace std;

int val[11][101];
int prv[11][101];
long long f[11][101];

int main() {
	// freopen("../uva116.in", "r", stdin);
	int m, n;

	while(cin >> m >> n) {
		for (int i = 1; i <= m; i++) {
			for (int j = n; j >= 1; j--) {
				cin >> val[i][j];
			}
		}

		// for (int i = 1; i <= m; i++) {
		// 	for (int j = 1; j <= n; j++) {
		// 		cout << val[i][j] << ' ';
		// 	}
		// 	cout << '\n';
		// }

		for (int i = 1; i <= m; i++) {
			f[i][1] = val[i][1];
		}

		for (int j = 2; j <= n; j++) {
			for (int i = 1; i <= m; i++) {
				f[i][j] = INT_MAX;
				prv[i][j] = INT_MAX;
				for (int offset = -1; offset <= 1; offset++) {
					int k = i+offset;
					if (k == 0) k = m;
					else if (k == m+1) k = 1;

					if (f[k][j-1] + val[i][j] < f[i][j]) {
						// if(i==1&&j==2) cout << "debug: " << k << '\t' << j << '\t' << f[k][j-1] << '\t' << val[i][j] << '\t' << f[i][j] << '\n';
						f[i][j] = f[k][j-1] + val[i][j];
						prv[i][j] = k;
					} else if (f[k][j-1] + val[i][j] == f[i][j]) {
						prv[i][j] = min(prv[i][j], k);	
					}
				}
			}
		}

		// for (int i = 1; i <= m; i++) {
		// 	for (int j = 1; j <= n; j++) {
		// 		cout << f[i][j] << '\t';
		// 	}
		// 	cout << '\n';
		// }

		int min_row = -1;
		long long min_val = INT_MAX;
		// 回溯最佳路径
		for (int i = 1; i <= m; i++) {
			if (f[i][n] < min_val) {
				min_val = f[i][n];
				min_row = i;
			}
		}

		vector<int> rev_path = {min_row,};
		for (int j = n; j >= 2; j--) {
			rev_path.push_back(prv[min_row][j]);
			min_row = prv[min_row][j];
		}
		for (size_t l = 0; l < rev_path.size(); l++) {
			printf("%d", rev_path[l]);
			if (l != rev_path.size()-1) printf(" ");
		}
		printf("\n%lld\n", min_val);
	}

	return 0;
}

/*
此题难点在于字典序；因此最后循迹只能找后继，而非前驱
*/