# algo-know-yeah Team Note

| ID | 이름 |
|--|--|
|singun11|김신건|
|antifly55|서형빈|
|Neogulee|최성훈|

## Table of contents

0. Base

1. Graph
    1. dijkstra
    2. bellman-ford
    3. kruskal
    4. prim
    5. topological sort
    6. . union-find

2. Tree
    1. segment tree
    2. segment tree with lazy propagation

3. String
    1. KMP

4. Extra
    1. Treap 
    2. MCC

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

### 1.2. bellman-ford
```cpp

```
### 1.3. kruskal
```cpp

```
### 1.4. prim
```cpp

```
### 1.5. topological sort 
```cpp

```

### 1.6. union-find
```cpp

```

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