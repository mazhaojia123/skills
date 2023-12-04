// 贪心；不被罚就是赚；每天总是做赚的最多的；
#include <iostream>
#include <algorithm>
using namespace std;

struct Job {
	int id;
	double t, s;	
};
Job jobs[1001];

bool cmp_g(const Job &a, const Job& b) {
	if (a.s/a.t - b.s/b.t > 1e-5) return true;
	else if(a.s/a.t - b.s/b.t < 1e-5) return false;
	else return a.id < b.id;
}

int main() {
	freopen("../uva10026.in", "r", stdin);
	int c_num, n;
	cin >> c_num;

	for (int c_cnt = 0; c_cnt < c_num; c_cnt++) {
		cin >> n;
		for (int i = 1; i <= n; i++) {
			double tmp_t, tmp_s;
			cin >> tmp_t >> tmp_s;
			jobs[i].id = i;
			jobs[i].s = tmp_s;
			jobs[i].t = tmp_t;
		}
		sort(jobs+1, jobs+1+n, cmp_g);
		for (int i = 1; i <= n-1; i++) {
			printf("%d ", jobs[i].id);
		}
		printf("%d\n", jobs[n].id);
		if (c_cnt < c_num-1)
			printf("\n");
	}

}