#include <iostream>
#include <string>
using namespace std;

int cnt[26];
int check(const string &s) {
	cnt[s[0]-'a']++;
	cnt[s[1]-'a']++;
	cnt[s[2]-'a']++;
	cnt[s[3]-'a']++;
	int freq = 0; 
	for (int i = 0; i < 26; i++) {
		if (cnt[i] > 0)	{
			freq++;
		}
	}
	if (freq == 4) return 3;
	else if (freq == 3) return 2;
	else if (freq == 2) return 1;
	else if (freq == 1) return 0;
	return -1;
}
void init() {
	for (int i = 0; i < 26; i++) cnt[i] = 0;
}

int main() {
	freopen("../codeforces1721A.in", "r", stdin);
	int t;
	cin >> t;
	for (int t_cnt = 0; t_cnt < t; t_cnt++) {
		string s1, s2;
		cin >> s1 >> s2;
		s1 = s1+s2;
		init();
		printf("%d\n", check(s1));
	}
}