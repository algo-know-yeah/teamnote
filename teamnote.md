# algo-know-yeah Team Note

| | |
|--|--|
|singun11|김신건|
|antifly55|서형빈|
|loes353|최성훈|

## Table of contents

0. Base

1. Graph
    1. dijkstra
    2. 

2. String
    1. KMP
    2. 
3. 

## 0. Base
```cpp
#include <bits/stdc++.h>

#define for1(s,n) for(int i = s; i < n; i++)
#define for1j(s,n) for(int j = s; j < n; j++)
#define foreach(k) for(auto i : k)
#define foreachj(k) for(auto j : k)
#define pb(a) push_back(a)
#define sz(a) a.size()

using namespace std;
typedef unsigned long long ull;
typedef long long ll;
typedef vector <int> iv1;
typedef vector <vector<int>> iv2;
typedef vector <ll> llv1;
typedef unsigned int uint;
typedef vector <ull> ullv1;
typedef vector <vector <ull>> ullv2;

int main() {
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);	
}
```

## 1. Graph
### 1.1. dijkstra
```cpp
#define MAX 100000
#define INF (ll)1e18

struct edge {
	int node;
	ll cost;
	bool operator<(const st &to) const {
		return cost > to.cost;
	}
};

ll dist[MAX_V + 1];
vector<edge> adj[MAX_V + 1];

ll dijkstra(int n,int start) {
	fill(dist, dist + n + 1, INF);
	priority_queue<edge> pq;
	pq.push({ start, 0 });
	dist[start] = 0;
	while (!pq.empty()) {
		edge cur = pq.top();
		pq.pop();

		if (cur.cost > dist[cur.node]) continue;

		for (st &nxt : adj[cur.node])
			if (dist[cur.node] + nxt.cost < dist[nxt.node]) {
				dist[nxt.node] = dist[cur.node] + nxt.cost;
				pq.push({ nxt.node, dist[nxt.node] });
			}
	}
	return dist[n] >= INF ? -1 : dist[n];
}
```

### 1.2. 

## 2. String

### 2.1. KMP

```cpp
string content;
string obj;
int fail[MX];

vector <int> kmp (string s, string o) {
    fill(fail,fail+MX,0);
    vector<int> result;
    int N = s.length();
    int M = o.length();
    for(int i=1, j=0; i<M; i++){
        while(j > 0 && o[i] != o[j]) j = fail[j-1];
        if(o[i] == o[j]) fail[i] = ++j;
    }
    for(int i = 0, j = 0; i < N; i++) {
        while(j > 0 && s[i] != o[j]) j = fail[j-1];
        if(s[i] == o[j]) {
            if(j == M-1) { // matching OK;
                result.push_back(i - M + 2);
                j = fail[j];
            }
            else j++;
        }
    }
    return result;
}
```