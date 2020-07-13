#include <bits/stdc++.h>

#define for1(s,n) for(int i = s; i < n; i++)
#define MAX 1000100

using namespace std;
typedef long long ll;

ll seg[4 * MAX], lazy[4 * MAX];

void update_lazy(ll node, ll x, ll y) {
    if (!lazy[node])
        return;
    seg[node] += (y - x + 1)*lazy[node];
    if (x != y) {
        lazy[node * 2] += lazy[node];
        lazy[node * 2 + 1] += lazy[node];
    }
    lazy[node] = 0;
}

ll update(ll lo, ll hi, ll val, ll node, ll x, ll y) {
    update_lazy(node, x, y);
    if (y < lo || hi < x)
        return seg[node];
    if (lo <= x && y <= hi) {
        lazy[node] += val;
        update_lazy(node, x, y);
        return seg[node];
    }
    ll mid = (x + y)/2;
    return seg[node] = update(lo, hi, val, node * 2, x, mid) + update(lo, hi, val, node * 2 + 1, mid + 1, y);
}

ll query(ll lo, ll hi, ll node, ll x, ll y) {
    update_lazy(node, x, y);
    if (y < lo || hi < x)
        return 0;
    if (lo <= x && y <= hi)
        return seg[node];
    ll mid = (x + y)/2;
    return query(lo, hi, node * 2, x, mid) + query(lo, hi, node * 2 + 1, mid + 1, y);
}

int main() {
	ios::sync_with_stdio(false); 
	cin.tie(NULL); 
	cout.tie(NULL); 
	
	ll a, b, c, d, n, m, k;
	
    cin >> n >> m >> k;
    for1(1, n+1) {
        cin >> a;
        update(i, i, a, 1, 1, n);
    }
    for1(0, m+k) {
        cin >> a;
        if (a == 1) {
            cin >> b >> c >> d;
            update(b, c, d, 1, 1, n);
        }
        else {
            cin >> b >> c;
            cout << query(b, c, 1, 1, n) << endl;
        }
    }
    return 0;
}
