#include <iostream>
#include <vector>
#define N 31

using namespace std;

struct Base {
	int l, w, height;
	Base(int x, int y, int hei) {
		l = x;
		w = y;
		height = hei;
	}
};

void swap(int &x, int &y) {
	int tmp = x;
	x = y;
	y = tmp;
}

bool fit(const Base &lower, const Base &upper) {
	return upper.l < lower.l && upper.w < lower.w;
}


int n, x, y, z;
int f[3*N][3*N];

int main() {
	// freopen("../uva437.in", "r", stdin);
	int cnt = 0;
	while (cin >> n, n) {
		cnt += 1;
		vector<Base> bases = {Base(0,0,0), };
		for (int i = 0; i < n; i++)	{
			cin >> x >> y >> z;
			// printf("%d %d %d\n", x, y, z);
			bases.push_back(Base(x,y,z));
			bases.push_back(Base(x,z,y));
			bases.push_back(Base(z,y,x));
			int p = bases.size() - 1;
			if (bases[p].l < bases[p].w) { swap(bases[p].l, bases[p].w); }
			if (bases[p-1].l < bases[p-1].w) { swap(bases[p-1].l, bases[p-1].w); }
			if (bases[p-2].l < bases[p-2].w) { swap(bases[p-2].l, bases[p-2].w); }
		}
		// for (auto it : bases) {
		// 	printf("l:%d w:%d height:%d\n", it.l, it.w, it.height);
		// }

		for (int i = 0; i <= 3*n; i++) { f[1][i] = bases[i].height; }

		// for (int k = 1; k <= 3*n; k++) {
		// 	for (int i = 1; i <= 3*n; i++) {
		// 		printf("%d\t", f[k][i]);
		// 	}
		// 	printf("\n");
		// }


		for (int k = 2; k <= 3*n; k++) {
			for (int i = 1; i <= 3*n; i++) {
				f[k][i] = f[k-1][i];
				for (int j = 1; j <= 3*n; j++) {
					if (fit(bases[j], bases[i])) {
						f[k][i] = max(f[k][i], f[k-1][j] + bases[i].height);
					}
				}
			}
		}

		int max_height = -1;
		for (int i = 1; i <= 3*n; i++) {
			max_height = max(max_height, f[3*n][i]);	
		}

		printf("Case %d: maximum height = %d\n", cnt, max_height);
	}

	return 0;
}