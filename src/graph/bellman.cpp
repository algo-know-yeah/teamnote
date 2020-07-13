#include <bits/stdc++.h>

#define for1(s,n) for(int i = s; i < n; i++)
#define for1j(s,n) for(int j = s; j < n; j++)
#define pb(a) push_back(a)
#define sz(a) a.size()

#define MAX 100010
#define INF (ll)1e18

using namespace std;
typedef long long ll;

struct edge {
	int to, cost;
};

vector<edge> v[MAX];
ll D[MAX];
int n;

bool bellman(){
	bool isCycle = false;
	for1(1, n+1) {
		for1j(1, n+1) {
		for(int k=0; k<sz(v[j]); k++) {
			edge p = v[j][k];
			int end = p.to;
			ll dist = D[j] + p.cost;
				if (D[j] != INF && D[end] > dist) {
					D[end] = dist;
					if (i == n) isCycle = true;
				}
			}
		}
	}
	return isCycle;
}

int main() {
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);

	int m;
	int start, end, cost;

	cin >> n >> m;
	for1(1, n+1) D[i] = INF;
    
	// start point is 1
	D[1] = 0;
	for1(1, m+1) {
		cin >> start >> end >> cost;
		v[start].push_back({end, cost});
	}

	// output
	if (bellman()) {
		cout << -1 << '\n';
		return 0;
	}
	for (int v = 2; v <= n; ++v) {
		ll ans = (D[v] == INF) ? -1 : D[v];
		cout << ans << '\n';
	}
	return 0;
}
