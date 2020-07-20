#include <iostream>
#include <string>

using namespace std;

const int ALPHABETS = 26;

int chToIdx(char ch) { return ch - 'a'; }
struct Trie {
	bool check = false;
	Trie* chil[ALPHABETS];
	Trie() {
		for (int i = 0; i < ALPHABETS; i++)
			chil[i] = NULL;
	}
	~Trie() {
		for (int i = 0; i < ALPHABETS; i++)
			if (chil[i])
				delete chil[i];
	}
	void insert(string& s, int idx = 0) {
		if (idx == s.size()) {
			check = true;
			return;
		}
		int next = chToIdx(s[idx]);
		if (chil[next] == NULL)
			chil[next] = new Trie();
		chil[next]->insert(s, idx + 1);
	}
	bool find(string& s, int idx = 0) {
		if (idx == s.size())
			return check;
		int next = chToIdx(s[idx]);
		if (chil[next] == NULL)
			return false;
		return chil[next]->find(s, idx + 1);
	}
};

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);

	int N, M;
	cin >> N >> M;
	Trie trie;
	for (int i = 0; i < N; i++)
	{
		string s;
		cin >> s;
		trie.insert(s);
	}
	int ret = 0;
	for (int i = 0; i < M; i++)
	{
		string s;
		cin >> s;
		if (trie.find(s))
			ret++;
	}
	cout << ret << endl;
}
