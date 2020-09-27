#include <bits/stdc++.h>
using namespace std;

const double PI = acos(-1);
typedef complex<double> cpx;
 
void FFT(vector<cpx> &f, cpx w){
    int n = f.size();
    if(n == 1) return; 
 
    vector<cpx> even(n/2), odd(n/2);
    for(int i = 0; i < n; ++i)
        (i%2 ? odd : even)[i/2] = f[i];
 
    FFT(even, w*w);
    FFT(odd, w*w);
 
    cpx wp(1, 0);
    for(int i = 0; i < n/2; ++i){
        f[i] = even[i] + wp*odd[i];
        f[i + n/2] = even[i] - wp*odd[i];
        wp *= w;
    }
}

vector<cpx> multiply(vector<cpx> a, vector<cpx> b){
    int n = 1;
    while(n < a.size()+1 || n < b.size()+1) n *= 2;
    n *= 2;
    a.resize(n);
    b.resize(n);
    vector<cpx> c(n);
 
    cpx w(cos(2*PI/n), sin(2*PI/n));
 
    FFT(a, w);
    FFT(b, w);
 
    for(int i = 0; i < n; ++i)
        c[i] = a[i]*b[i];
 
    FFT(c, cpx(1, 0)/w);
    for(int i = 0; i < n; ++i){
        c[i] /= cpx(n, 0);
        c[i] = cpx(round(c[i].real()), round(c[i].imag()));
    }
    return c;
}

int main()
{
	ios::sync_with_stdio(false); 
	cin.tie(NULL); 
	cout.tie(NULL); 
	int i, count=0;
	string a, b;
	vector<cpx> av, bv, cv;
	vector<int> v;

	cin >> a >> b;
	if(a=="0" || b=="0"){
		cout << 0 << "\n";
		return 0;
	}
	for(i=0;i<a.size();i++) av.push_back(a[i]-'0');
	for(i=0;i<b.size();i++) bv.push_back(b[i]-'0');
	while(!av.empty() && av[av.size()-1].real()==0) {av.pop_back(); count++;}
	while(!bv.empty() && bv[bv.size()-1].real()==0) {bv.pop_back(); count++;}

	cv = multiply(av, bv);
	for(i=cv.size()-1;i>-1;i--){
		if(abs(cv[i].real()) != 0) break;
	}
	for(;i>-1;i--){
		v.push_back(cv[i].real());
	}

	for(i=0;i<v.size();i++){
		if(v[i]>=10 && i+1 == v.size()){
			v.push_back(v[i]/10);
			v[i] %= 10;
		}
		else{
			v[i+1] += v[i]/10;
			v[i] %= 10;
		}
	}

	for(i=v.size()-1;i>-1;i--) cout << v[i];
	for(i=0;i<count;i++) cout << 0;
	cout << "\n";

	return 0;
}