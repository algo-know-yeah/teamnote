struct LCA {
	vector<int> serials, no2se, se2no, loc; // length, loc의 index는 no
	vector<ll> length;
	SegmentTree *seg; // 최솟값 segment tree
	LCA(vector<vector<pair<int, ll>>> &edges) {
		int N = edges.size();
		no2se = vector<int>(N, -1);
		se2no = vector<int>(N, -1);
		loc = vector<int>(N, -1);
		length = vector<ll>(N, -1);
		length[0] = 0;
		vector<bool> visited(N, false);

		init_serials(0, visited, edges);
		seg = new SegmentTree(serials);
		seg->init(1, 1, serials.size());
	}

	void init_serials(int current, vector<bool>& visited, vector<vector<pair<int, ll>>>&edges) {
		static int cnt = 0;
		visited[current] = true;
		if (no2se[current] == -1) {
			no2se[current] = cnt++;
			se2no[no2se[current]] = current;
			loc[current] = serials.size();
		}
		serials.push_back(no2se[current]);
		for1(0, edges[current].size()) {
			int next = edges[current][i].first;
			int cost = edges[current][i].second;
			if (visited[next])
				continue;
			length[next] = length[current] + cost;
			init_serials(next, visited, edges);
			serials.push_back(no2se[current]);
		}
		visited[current] = false;
	}
	
	ll query(int u, int v) { // 두 정점 사이의 거리
		if (loc[u] > loc[v])
			swap(u, v);
		ll lca = seg->query(loc[u] + 1, loc[v] + 1, 1, 1, serials.size());
		return length[u] + length[v] - 2ll * length[se2no[lca]];
	}
};

int main(void)
{
	vector<vector<pair<int, ll>>> edges;

	int N;
	cin >> N;
	edges.resize(N);
	for (int i = 0; i < N - 1; i++)
	{
		int a, b, c;
		scanf("%d %d %d", &a, &b, &c);
		a--; b--;
		edges[a].push_back({ b, c });
		edges[b].push_back({ a, c });
	}

	LCA lca(edges);

	int M;
	cin >> M;
	for (int i = 0; i < M; i++)
	{
		int u, v;
		scanf("%d %d", &u, &v);
		u--; v--;
		printf("%d\n", lca.query(u, v));
	}
}
