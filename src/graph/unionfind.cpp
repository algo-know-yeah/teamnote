#include <bits/stdc++.h>
#define MAXN 100010

#define for1(s,n) for(int i = s; i < n; i++)

using namespace std;

int root[MAXN];
int level[MAXN];

void init(int n) {
	for1(0, n){
		root[i] = i;
		level[i] = 1;
	}
}

int find(int x) {
	return root[x] == x ? x : root[x] = find(root[x]);
}

void merge(int x, int y) {
	x = find(x);
	y = find(y);
	if (x == y) return;
	if (level[x] < level[y]) root[x] = y;
	else root[y] = x;
	if (level[x] == level[y]) level[x]++;
}

int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);	
	
	int n, m, a, b;
	cin >> n >> m;
	
	init(n);
	for1(0, m){
		cin >> a >> b;
		merge(a, b);
	}
	
	for1(0, n){
		cout << root[i] << " ";
	}
	
	return 0;
}
