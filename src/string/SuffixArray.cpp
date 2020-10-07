#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 알고리즘 문제 해결 전략

struct SuffixComparator {
	const vector<int>& group;
	int t;
	SuffixComparator(const vector<int>& _group, int _t) :group(_group), t(_t) { }
	bool operator() (int a, int b) {
		if (group[a] != group[b])
			return group[a] < group[b];
		return group[a + t] < group[b + t];
	}
};
vector<int> getSuffixArr(const string& s) {
	int n = s.size();
	int t = 1;
	vector<int> group(n + 1);
	for (int i = 0; i < n; i++) group[i] = s[i];
	group[n] = -1;
	vector<int> perm(n);
	for (int i = 0; i < n; i++) perm[i] = i;
	while (t < n) {
		SuffixComparator compare(group, t);
		sort(perm.begin(), perm.end(), compare);
		t *= 2;
		if (t >= n) break;
		vector<int> new_group(n + 1);
		new_group[n] = -1;
		new_group[perm[0]] = 0;
		for (int i = 1; i < n; i++)
			if (compare(perm[i - 1], perm[i]))
				new_group[perm[i]] = new_group[perm[i - 1]] + 1;
			else
				new_group[perm[i]] = new_group[perm[i - 1]];
		group = new_group;
	}
	return perm;
}
int commonPrefix(const string& s, int i, int j) {
	int k = 0;
	while (i < s.size() && j < s.size() && s[i] == s[j]) {
		i++; j++; k++;
	}
	return k;
}
int countSubstrs(const string& s) { // 부분 문자열의 개수
	vector<int> a = getSuffixArr(s);
	int ret = 0;
	int n = s.size();
	for (int i = 0; i < a.size(); i++) {
		int cp = 0;
		if (i > 0) cp = commonPrefix(s, a[i - 1], a[i]);
		ret += s.size() - a[i] - cp;
	}
	return ret;
}
int longestFrequent(int k, const string& s) { // k 번 이상 등장하는 부분 문자열 중 최대 길이
	vector<int> a = getSuffixArr(s);
	int ret = 0;
	for (int i = 0; i + k <= s.size(); i++)
		ret = max(ret, commonPrefix(s, a[i], a[i + k - 1]));
	return ret;
}
// 최장 중복 부분 문자열의 길이
// Kasai, T. et al., "Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its Applications"
int getHeight(const string& s, vector<int>& pos)
{
	const int n = pos.size();
	vector<int> rank(n);
	for (int i = 0; i < n; i++)
		rank[pos[i]] = i;
	int h = 0, ret = 0;
	for (int i = 0; i < n; i++)
	{
		if (rank[i] > 1) {
			int j = pos[rank[i] - 1];
			while (s[i + h] == s[j + h])
				h++;
			ret = max(ret, h);
			if (h > 0)
				h--;
		}
	}
	return ret;
}
int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);

	string str;
	cin >> str;
	vector<int> arr = getSuffixArr(str);
	
	for (int i = 0; i < str.size(); i++)
		cout << arr[i] << "\n";
	cout << countSubstrs(str) << endl;
}
