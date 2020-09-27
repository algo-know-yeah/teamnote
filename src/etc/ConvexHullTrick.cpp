#include <bits/stdc++.h>
#define MAX 100010
using namespace std;

typedef long long ll;
struct linear{
	ll a, b;
	double s;
};

ll dp[MAX], top=0;
linear f[MAX];

double cross(linear &f, linear &g){
	return (g.b-f.b)/(f.a-g.a);
}

void addLine(ll a, ll b){ // y = ax + b
    linear g({a, b, 0});
    while(top > 0){
        g.s = cross(f[top-1], g);
        if(f[top-1].s < g.s) break;
        top--;
    }
    f[top++] = g;
}

ll searchLine(ll x){
    ll pos = top-1;
    if(x < f[top-1].s){
        ll lo = 0, hi = top-1;
        while(lo+1 < hi){
            ll mid = (lo+hi)/2;
            (x < f[mid].s ? hi:lo) = mid;
        }
        pos = lo;
    }
    return pos;
}

int main(){
    ll n, i, a[MAX], b[MAX];

	cin >> n;
	for(i=0;i<n;i++) cin >> a[i];
	for(i=0;i<n;i++) cin >> b[i];

	for(i=1;i<n;i++){
		addLine(b[i-1], dp[i-1]);
        ll pos = searchLine(a[i]);
		dp[i] = f[pos].a * a[i] + f[pos].b;
	}

	cout << dp[n-1] << "\n";

    return 0;
}