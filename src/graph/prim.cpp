#include <bits/stdc++.h>

#define MAX 100010
#define INF (ll)1e18

#define for1(s,n) for(int i = s; i < n; i++)
#define pb(a) push_back(a)
#define sz(a) a.size()

using namespace std;
typedef long long ll;

struct edge{
	int node, cost;
};

int V;
vector<pair<int, int> > graph[MAX];
 
ll prim(vector<pair<int, int> > &selected) {
    selected.clear();
 
    vector<bool> added(V, false);
    vector<ll> minWeight(V, INF), parent(V, -1);
 
    ll ret = 0;
    minWeight[0] = parent[0] = 0;
    for (int iter = 0; iter < V; iter++) {
        int u = -1;
        for (int v = 0; v < V; v++) {
            if (!added[v] && (u == -1 || minWeight[u]>minWeight[v]))
                u = v;
        }
 
        if (parent[u] != u)
            selected.push_back(make_pair(parent[u], u));
 
        ret += minWeight[u];
        added[u] = true;

		for1(0, sz(graph[u])) {
            int v = graph[u][i].first, weight = graph[u][i].second;
            if (!added[v] && minWeight[v]>weight) {
                parent[v] = u;
                minWeight[v] = weight;
            }
        }
    }
    return ret;
}
 
int main() {
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
	
	int m, start, end, cost;
	cin >> V >> m;
    for1(0, m){
    	cin >> start >> end >> cost;
    	graph[start].push_back({end, cost});
    	graph[end].push_back({start, cost});
    }
 
    vector<pair<int, int> > selected;
    int mst=prim(selected);
    
    printf("mst:%d\n", mst);
    for1(0, sz(selected)) {
        cout << selected[i].first << "-" << selected[i].second << "\n";
    }
}
