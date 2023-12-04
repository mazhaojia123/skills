#include <iostream>
#include <algorithm>

using namespace std;

int m, n;
int h_d[20005], h_k[20005];

int main() {
	// freopen("../uva11292.in", "r", stdin);
	while(cin >> n >> m, m||n) {
		for (int i = 0; i < n; i++) 
			cin >> h_d[i];
		for (int i = 0; i < m; i++) 
			cin >> h_k[i];
		
		sort(h_d, h_d+n);
		sort(h_k, h_k+m);
		
		// for (int i = 1; i <= n; i++) 
		// 	cout <<  h_d[i] << ' ';
		// cout << endl;
		// for (int i = 1; i <= m; i++) 
		// 	cout <<  h_k[i] << ' ';
		// cout << endl;
		int cur=0;
		int count=0;
		int i=0;
		for(;i<n;i++,cur++){
			while(h_d[i]>h_k[cur]&&cur<m)cur++;
			if(cur>=m&&i<n){
				printf("Loowater is doomed!\n");
				break;
			}
			count+=h_k[cur];
		}
		if(i==n)printf("%d\n",count);
		
	}
}