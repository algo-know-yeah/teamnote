#include <iostream>
#define p 1000000007
using namespace std;

typedef long long ll;
ll pow(ll a, ll b){
	if(b == 0) return 1;
	ll n = pow(a, b/2)%p;
	ll temp = (n * n)%p;
	
	if(b%2==0) return temp;
	return (a * temp)%p;
}

ll fermat(ll a, ll b){
	return a%p*pow(b, p-2)%p;
}

int main()
{
	cout << fermat(100, 20) << endl;
	// 페르마의 소정리로 (a/b)%p 를 계산함
	
	return 0;
}
