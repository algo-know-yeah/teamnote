#include <bits/stdc++.h>
#define MAX 100010
using namespace std;

int ans, cnt;
int visit[MAX], sn[MAX];
bool finished[MAX];
vector<int> adj[MAX];
stack<int> st;
vector<vector<int> > scc;

int dfs(int curr){
    visit[curr] = ++cnt;
    st.push(curr);
 
    int result = visit[curr];
    for(int i = 0; i < adj[curr].size(); i++){
        int next = adj[curr][i];
        if(visit[next] == 0) result = min(result, dfs(next));
        else if(!finished[next]) result = min(result, visit[next]);
    }
 
    if(result == visit[curr]){
        vector<int> currSCC;
        while(1){
            int t = st.top();
            st.pop();
            currSCC.push_back(t);
            finished[t] = true;
            sn[t] = ans;
            if(t == curr) break;
        }
        sort(currSCC.begin(), currSCC.end());
        scc.push_back(currSCC);
        ans++;
    }
    return result;
}

void makeSCC(int v){
    for(int i=1; i<=v; i++)
        if(!visit[i]) dfs(i);
    sort(scc.begin(), scc.end());
}

int main(){
    int start, end, i, j, v, e;
    cin >> v >> e;
    for(i=0; i<e; i++){
        cin >> start >> end;
        adj[start].push_back(end);
    }
 
    makeSCC(v);
 
    cout << ans << "\n";
    for(int i=0; i<scc.size(); i++){
        for(int j=0; j<scc[i].size(); j++)
            cout << scc[i][j] << " ";
        cout << -1 << "\n";
    }
}