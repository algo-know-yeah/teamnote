#include <bits/stdc++.h> 
#define MAX 5001
#define INF 987654321
using namespace std;

int dp[MAX][MAX], K[MAX][MAX];
int arr[MAX];
int sum[MAX];
 
int solve(int n) {
    for (int m = 2; m <= n; m++) {
        for (int i = 0; m + i <= n; i++) {
            int j = i + m;
            for (int k = K[i][j - 1]; k <= K[i + 1][j]; k++) {
                int now = dp[i][k] + dp[k][j] + sum[j] - sum[i];
                if (dp[i][j] > now)
                    dp[i][j] = now, K[i][j] = k;
            }
        }
    }
    return dp[0][n];
}
 
int main() {
 
    ios::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);
 
    int t;
    cin >> t;
 
    while (t--) {
        int n;
        cin >> n;
        fill(&dp[0][0], &dp[MAX-1][MAX-1], INF);
        for (int i = 1; i <= n; i++)
            cin >> arr[i], sum[i] = sum[i - 1] + arr[i], K[i - 1][i] = i, dp[i - 1][i] = 0;
        cout << solve(n) << "\n";
    }
    return 0; 
}

/*
if
C[a][c] + C[b][d] <= C[a][d] + C[b][c] (a<=b<=c<=d)
C[b][c] <= C[a][d] (a<=b<=c<=d)

then
dp[i][j] = min(dp[i][k] + dp[k][j]) + C[i][j]
range of k: A[i, j-1] <= A[i][j]=k <= A[i+1][j]
*/

// 파일 합치기 2 코드입니다.  
