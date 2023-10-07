#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
	string s1, s2, s3, res;
	cin >> s1 >> s2;
	int len = min(s1.length(), s2.length());

	for (int i = 0; i < len; i++) {
		res += s1[i];
		res += s2[i];
	}

	if (s1.length() < s2.length()) {
		s3 = s2;
	} else {
		s3 = s1;
	}

	for (int i = len; i < s3.length(); i++) {
		res += s3[i];
	}
	pow	
	cout << res;
}