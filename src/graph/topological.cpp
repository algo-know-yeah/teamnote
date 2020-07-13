#include <bits/stdc++.h>
#define MAXN 100010

#define for1(s,n) for(int i = s; i < n; i++)
#define pb(a) push_back(a)
#define sz(a) a.size()

using namespace std;

typedef vector <int> iv1;

int n; 
int link[MAXN];
iv1 graph[MAXN];

void topologySort() {
	iv1 result;
	queue<int> q;
	
	for1(1, n+1) {
		if(link[i] == 0) q.push(i);
	}

	while(!q.empty()) {
		int x = q.front();
		q.pop();
		result.pb(x);

		for1(0, sz(graph[x])) {
			int y = graph[x][i];
			if(--link[y]==0) q.push(y);
		}
	}

	for1(0, n) {
		cout << result[i] << " ";
	}
 
}

int main() {
	int m, start, end;
	
	cin >> n >> m;
	for1(0, m){
		cin >> start >> end;
		graph[start].pb(end);
		link[end]++;
	}

	topologySort();
	return 0;
}
