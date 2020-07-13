#include <bits/stdc++.h>

#define for1(s,n) for(int i = s; i < n; i++)
#define for1j(s,n) for(int j = s; j < n; j++)
#define pb(a) push_back(a)
#define sz(a) a.size()

using namespace std;

double getR(double x, double y){
	return x*x + y*y;
}

double avg(vector<double> x){
	double ans=0;
	for(int i=0; i<sz(x); i++) ans+=x[i];
	return ans/sz(x);
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

	double inputx, inputy, rx, ry, distance, lr=1;
	int n, index;
	vector<double> x, y;
	
	cin >> n;
	for1(0, n){
		cin >> inputx >> inputy;
		x.pb(inputx);
		y.pb(inputy);
	}
	
	rx = avg(x);
	ry = avg(y);
	
	for1(0, 100000){
		distance = -1; index = -1;
		for1j(0, n){
			if(distance < getR(x[j] - rx, y[j] - ry)){
				distance = getR(x[j] - rx, y[j] - ry);
				index = j;
			}
		}
		rx = rx + (x[index] - rx) * lr;
		ry = ry + (y[index] - ry) * lr;
		lr *= 0.999;
	}
    
	cout << fixed;
	cout.precision(2);
	cout << sqrt(distance) << endl;
	
	return 0;
}
