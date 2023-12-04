#include <iostream>
#include <algorithm>

using namespace std;

struct Dragon {
	int x,y;
};
Dragon dragons[1001];

bool less_dragon(const Dragon &a, const Dragon &b) {
	return a.x < b.x;
}

int main() {
	// freopen("../codeforces230A.in", "r", stdin);
	int s, n;	
	cin >> s >> n;
	for (int i = 0; i < n; i++) {
		cin >> dragons[i].x >> dragons[i].y;
	}
	sort(dragons, dragons+n, less_dragon);
	bool pass = true;
	for (int i = 0; i < n; i++) {
		if (s <= dragons[i].x) {
			pass = false;
			break;
		} else {
			s += dragons[i].y;
		}
	}
	if (pass) printf("YES\n");
	else printf("NO\n");
}