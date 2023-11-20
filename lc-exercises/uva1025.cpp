#include <iostream>
#include <cstring>

using namespace std;

int N, T;
int t_gap[55];
int M1, M2;
bool from_right[205][55], from_left[205][55];
int f[205][55];

int main() {
	freopen("../uva1025.in", "r", stdin);
	int cnt = 0;
	while(cin >> N, N) {
		cnt++;
		memset(from_right, 0, sizeof(from_right));
		memset(from_left, 0, sizeof(from_left));
		memset(f, 0x3f, sizeof(f));

		cin >> T;
		for (int i = 1; i <= N-1; i++) {cin >> t_gap[i];}

		cin >> M1;
		for (int k = 1; k <= M1; k++) {
			int cur_time;
			cin >> cur_time;
			for (int j = 2; j <= N; j++) {
				cur_time += t_gap[j-1];
				if (cur_time > T) break;
				from_left[cur_time][j] = true;
			}
		}

		cin >> M2;
		for (int k = 1; k <= M2; k++) {
			int cur_time;
			cin >> cur_time;
			for (int j = N-1; j >= 1; j--) {
				cur_time += t_gap[j];
				if (cur_time > T) break;
				from_right[cur_time][j] = true;
			}
		}
		

		f[0][1] = 0;
		for (int i = 1; i <= T; i++) {
			for (int j = 1; j <= N; j++) {
				// f[i][j] = 0x3f3f3f3f;
				f[i][j] = f[i-1][j]+1;
				if (from_left[i][j]) { f[i][j] = min(f[i][j], f[i-t_gap[j-1]][j-1]); }
				if (from_right[i][j]) { f[i][j] = min(f[i][j], f[i-t_gap[j]][j+1]); }
			}
		}

		if (f[T][N] >= 0x3f3f3f3f) {
			printf("Case Number %d: impossible\n", cnt);
		} else {
			printf("Case Number %d: %d\n", cnt, f[T][N]);
		}
	}	

	return 0;
}