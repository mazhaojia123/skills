#include <bits/stdc++.h>
using namespace std;

int N, V;
int v[1001], w[1001];
int f[1001][1001];

int main() {
	freopen("../acwing2.in", "r", stdin);

	cin >> N >> V;
	for (int i = 1; i <= N; i++) { cin >> v[i] >> w[i]; }

	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= V; j++) {
			f[i][j] = f[i-1][j];
			if(j >= v[i]) { f[i][j] = max(f[i][j], f[i-1][j-v[i]]+w[i]); }
		}
	}

	int res = -1;
	for (int j = 1; j <= V; j++) {
		res = max(res, f[N][j]);
	}
	printf("%d", res);
}