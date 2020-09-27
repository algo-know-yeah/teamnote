#include <bits/stdc++.h>

#define for1(s,n) for(int i = s; i < n; i++)
#define MAX 100010

using namespace std;

typedef long long ll;

struct SegmentTree {
	ll a[MAX], tree[MAX * 4];
	
	SegmentTree(vector<int> &vec) {
		for1 (0, vec.size())
			a[i + 1] = vec[i];
	}

	void init(int node, int x, int y) {
		if (x == y) {
			tree[node] = a[x];
			return;
		}
		int mid = (x + y) / 2;
		init(node * 2, x, mid);
		init(node * 2 + 1, mid + 1, y);
		tree[node] = tree[node * 2] + tree[node * 2 + 1];
	}

	void update(int pos, ll val, int node, int x, int y) {
		if (pos < x || pos > y) return;
		if (x == y) {
			tree[node] = val;
			return;
		}
		int mid = (x + y) / 2;
		update(pos, val, node * 2, x, mid);
		update(pos, val, node * 2 + 1, mid + 1, y);
		tree[node] = tree[node * 2] + tree[node * 2 + 1];
	}

	ll query(int lo, int hi, int node, int x, int y) {
		if (lo > y || hi < x) return 0;
		if (lo <= x && y <= hi) return tree[node];
		int mid = (x + y) / 2;
		return query(lo, hi, node * 2, x, mid) + query(lo, hi, node * 2 + 1, mid + 1, y);
	}
};

int main() {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);

	int n, q;
	cin >> n >> q;
	vector<int> vec(n);
	for1(0, n)
		cin >> vec[i];
	SegmentTree seg(vec);
	seg.init(1, 1, n);

	while (q--) {
		int a, b, c, d;
		cin >> a >> b >> c >> d;
		int start = min(a, b);
		int end = max(a, b);
		cout << seg.query(start, end, 1, 1, n) << '\n';
		seg.update(c, d, 1, 1, n);
	}
	return 0;
}
