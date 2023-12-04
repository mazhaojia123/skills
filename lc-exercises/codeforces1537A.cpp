#include<iostream>

using namespace std;

int main() {
	// freopen("../codeforces1537A.in", "r", stdin);
	int t;
	cin >> t;
	for (int i = 0; i < t; i++) {
		int t_sum = 0, n, tmp;
		cin >> n;
		for (int j = 0; j < n; j++) {
			cin >> tmp;
			t_sum += tmp;
		}
		if (t_sum >= n) printf("%d\n", t_sum - n);
		else printf("1\n");
	}
}