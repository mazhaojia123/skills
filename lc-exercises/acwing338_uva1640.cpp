#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

/// @brief NOTE: 计算从l到r的几个位形成的数字的值是多少
int get(vector<int> num, int l, int r) {
	int res = 0; 
	for (int i = l; i >= r; i -- ){
		res = res* 10 + num[i];
	}
	return res;
}


int power10(int x) {
	int res = 1; 
	// NOTE: 这种计次的方法值得学习
	while(x --) {
		res *= 10;
	}
	return res;
}

// count : 从第 1 个数到第 n 个数，x 出现的次数
int count(int n, int x) {
	if (!n) return 0;

	vector<int> num;
	// NOTE: 拆分数位
	while(n) {
		num.push_back(n % 10);
		n /= 10;
	}

	n = num.size();
	int res = 0;
	// NOTE: !x 少写一个判断
	for (int i = n - 1 - !x; i >= 0; i--) {
		if (i < n - 1) {
			res += get(num, n-1, i+1) * power10(i);
			if (!x) res -= power10(i);
		}

		if (num[i] == x) {
			res += get(num, i-1, 0) + 1; 
		}
		else if (num[i] > x) {
			res += power10(i);
		}
	}

	return res;
}

void swap(int &a, int &b) {
	int tmp = a;
	a = b;
	b = tmp;
}

int main() {
	// freopen("../acwing338.in", "r", stdin);
	int a, b;
	// 1. 一个值得学习的写法
	while(cin >> a >> b, a || b) {  
		if (a > b) swap(a, b);
		for (int i = 0; i < 10; i++) {
			cout << count(b, i) - count(a - 1, i);
			if (i!=9) cout << ' ';
		}
		cout << endl;
	}
}