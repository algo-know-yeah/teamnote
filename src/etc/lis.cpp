#include <bits/stdc++.h>
using namespace std;

#define for1(s,n) for(int i = s; i < n; i++)
#define pb(a) push_back(a)
#define sz(a) a.size()

typedef vector <int> iv1;

void lis(){
	int n, i, x;
	iv1 v, buffer;
	iv1::iterator vv;
	vector<pair<int, int> > print;
	v.pb(2000000000);
	
	cin >> n;
	for1(0, n){
		cin >> x;
		if(x > *v.rbegin()) {
			v.pb(x);
			print.push_back({v.size()-1, x});
		}
		else{
			vv = lower_bound(v.begin(), v.end(), x);
			*vv = x;
			print.push_back({vv-v.begin(), x});
		}
	}
	cout << sz(v) << endl;
	
	for(i=sz(print)-1;i>-1;i--){
		if(print[i].first == sz(v)-1){
			buffer.pb(print[i].second);
			v.pop_back();
		}
	}
	for(i=sz(buffer)-1;i>-1;i--) cout << buffer[i] << " ";
}

int main()
{
	lis();
	
	return 0;
}
