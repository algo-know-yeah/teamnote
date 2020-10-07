#include <iostream>
#include <string>
#include <vector>
#include <queue>

using namespace std;

const int ALPHABETS = 26;

int chToIdx(char ch) { return ch - 'a'; }
struct Trie {
	int terminal = -1;
	Trie* fail;
	vector<int> output;
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
	void insert(string& s, int number, int idx) {
		if (idx == s.size()) {
			terminal = number;
			return;
		}
		int next = chToIdx(s[idx]);
		if (chil[next] == NULL)
			chil[next] = new Trie();
		chil[next]->insert(s, number, idx + 1);
	}
	int find(string& s, int idx = 0) {
		if (idx == s.size())
			return terminal;
		int next = chToIdx(s[idx]);
		if (chil[next] == NULL)
			return false;
		return chil[next]->find(s, idx + 1);
	}
};
void computeFail(Trie* root) {
	queue<Trie*> q;
	root->fail = root;
	q.push(root);
	while (!q.empty()) {
		Trie* here = q.front();
		q.pop();
		for (int i = 0; i < ALPHABETS; i++) {
			Trie* child = here->chil[i];
			if (!child)	continue;
			if (here == root)
				child->fail = root;
			else {
				Trie* t = here->fail;
				while (t != root && t->chil[i] == NULL)
					t = t->fail;
				if (t->chil[i]) t = t->chil[i];
				child->fail = t;
			}
			child->output = child->fail->output;
			if (child->terminal != -1)
				child->output.push_back(child->terminal);
			q.push(child);
		}
	}
}
vector<pair<int, int>> ahoCorasick(string& s, Trie* root) {
	vector<pair<int, int>> ret;
	Trie* state = root;
	for (int i = 0; i < s.size(); i++) {
		int idx = chToIdx(s[i]);
		while (state != root && state->chil[idx] == NULL)
			state = state->fail;
		if (state->chil[idx])
			state = state->chil[idx];
		for (int j = 0; j < state->output.size(); j++)
			ret.push_back({ i, state->output[j] });
	}
	return ret;
}
int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);

	int N, M;
	cin >> N;
	Trie trie;
	for (int i = 0; i < N; i++)
	{
		string s;
		cin >> s;
		trie.insert(s, i, 0);
	}
	computeFail(&trie);

	cin >> M;
	for (int i = 0; i < M; i++)
	{
		string s;
		cin >> s;
		if (!ahoCorasick(s, &trie).empty())
			cout << "YES" << "\n";
		else
			cout << "NO" << "\n";
	}
}
