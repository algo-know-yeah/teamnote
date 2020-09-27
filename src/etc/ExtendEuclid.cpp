#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> pii;

int gcd(int a, int b){
	if(b==0) return a;
	return gcd(b, a%b);
}

// ax+by=gcd(a,b)
pii ext_gcd(int a, int b){
	if(b==0) return pii(1, 0);
	pii tmp = ext_gcd(b, a%b);
	return pii(tmp.second, tmp.first - (a/b) * tmp.second);
}

// ax = 1 (mod b) 
ll mod_inv(int a, int b){
    return (ext_gcd(a, b).first + b) % b;
}

int main(){
	ios_base::sync_with_stdio(false);
	cout.tie(NULL); cin.tie(NULL);
	int t, k, c;

	cin >> t;
	while(t--){
		cin >> k >> c;
		if(gcd(k, c)!=1){
			cout << "IMPOSSIBLE\n";
			continue;
		}

		ll ans = mod_inv(c, k);
		while(c*ans <= k) ans += k;
		if(ans > 1e9) cout << "IMPOSSIBLE\n";
		else cout << ans << "\n";

	}

	return 0;
}