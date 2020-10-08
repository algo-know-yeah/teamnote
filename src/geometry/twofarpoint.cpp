#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long ull;
typedef long long ll;
typedef unsigned int uint;
typedef vector <ull> ullv1;
typedef vector <vector <ull>> ullv2;

struct point{
    ll x,y;
    ll p=0,q=0;
};

bool comp1(point a, point b) {
    if(a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

bool comp2 (point a, point b) {
    if(a.q * b.p != a.p*b.q)
        return a.q * b.p < a.p*b.q;
    return comp1(a,b);
}

ll ccw(point p1, point p2, point p3) {
    ll ret = (p1.x * p2.y + p2.x * p3.y + p3.x * p1.y - p2.x * p1.y - p3.x * p2.y - p1.x * p3.y);
    return ret >0?1:(ret<0?-1:0);
}

vector <ll> getConvexHull(vector <point> ar) {
    vector <ll> stk;
    stk.push_back(0);
    stk.push_back(1);
    int next = 2;
    while(next < ar.size()) {
        while(stk.size() >= 2) {
            int s = stk.back();
            stk.pop_back();
            int f = stk.back();
            if(ccw(ar[f],ar[s],ar[next]) > 0) {
                stk.push_back(s);
                break;
            }
        }
        stk.push_back(next++);
    }

    return stk; 
}


ll N;
point Z;

ll getDist(point p, point q){
    return (p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y);
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    vector <point> ar;

    cin >> N;
    for(int x=0; x<N; x++) {
        cin >> Z.x >> Z.y;
        ar.push_back(Z);
    }
    
    sort(ar.begin(),ar.end(),comp1);
    for(int x=1; x<N; x++) {
        ar[x].p = ar[x].x - ar[0].x;
        ar[x].q = ar[x].y - ar[0].y;
    }
    sort(ar.begin()+1,ar.end(),comp2);
    
    vector <ll> ret = getConvexHull(ar);

    int i, j=0;
    ll ans = 0;
    point p1, p2;
    for(i=0;i<ret.size();i++){
        int ni = (i+1)%ret.size();
        while(1){
            int nj = (j+1) % ret.size();
            int v = ccw({0, 0}, {ar[ret[ni]].x - ar[ret[i]].x, ar[ret[ni]].y - ar[ret[i]].y}, {ar[ret[nj]].x - ar[ret[j]].x, ar[ret[nj]].y - ar[ret[j]].y});
            if(v > 0) j = nj;
            else break;
        }
        ll v = getDist(ar[ret[i]], ar[ret[j]]);
        if(ans < v){
            ans = v;
            p1 = ar[ret[i]];
            p2 = ar[ret[j]];
        }
    }

    cout << fixed;
    cout.precision(9);
    cout << sqrt((double)ans) << "\n";
}