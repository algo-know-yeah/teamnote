// https://www.acmicpc.net/problem/13537

#include <bits/stdc++.h>

#define for1(s,n) for(int i = s; i < n; i++)
#define for1j(s,n) for(int j = s; j < n; j++)
#define foreach(k) for(auto i : k)
#define foreachj(k) for(auto j : k)
#define pb(a) push_back(a)
#define sz(a) a.size()

#define Mx 440000

using namespace std;
typedef unsigned long long ull;
typedef long long ll;
typedef vector <int> iv1;
typedef vector <vector<int>> iv2;
typedef vector <ll> llv1;
typedef unsigned int uint;
typedef vector <ull> ullv1;
typedef vector <vector <ull>> ullv2;

ll N,input,M,qa,qb,qk;

llv1 a;
llv1 mTree[Mx];
void makeTree(ll idx, ll ss, ll se) { 
    if(ss == se) { 
        mTree[idx].push_back(a[ss]); 
        return; 
    } 
  
    ll mid = (ss+se)/2;

    makeTree(2*idx+1, ss, mid); 
    makeTree(2*idx+2, mid+1, se); 
  
    merge(mTree[2*idx+1].begin(), mTree[2*idx+1].end(), mTree[2*idx+2].begin(), mTree[2*idx+2].end(), back_inserter(mTree[idx])); 
}

ll query(ll node, ll start, ll end, ll q_s, ll q_e, ll k) { 
    //i j k: Ai, Ai+1, ..., Aj로 이루어진 부분 수열 중에서 k보다 큰 원소의 개수를 출력한다.
    if (q_s > end || start > q_e) return 0; 
  
    if (q_s <= start && q_e >= end) { 
        return mTree[node].size() - (upper_bound(mTree[node].begin(), mTree[node].end(), k) - mTree[node].begin()); 
    } 
  
    ll mid = (start+end)/2; 
    ll p1 = query(2*node+1, start, mid, q_s, q_e, k); 
    ll p2 = query(2*node+2, mid+1, end, q_s, q_e, k); 
    return p1 + p2; 
} 

int main() 
{ 
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> N;
    for(ll x = 0; x< N; x++) {
        cin >> input;
        a.push_back(input);
    }
    makeTree(0, 0, N-1);
    cin >> M;
    for(ll x = 0; x<M; x++) {
        cin >> qa >> qb >> qk;
        if(qa > qb) {
            qa ^= qb;
            qb ^= qa;
            qa ^= qb;
        }
        cout <<query(0,0,N-1,qa-1,qb-1,qk) << "\n";
    }
    return 0; 
} 