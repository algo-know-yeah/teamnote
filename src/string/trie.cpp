#include <iostream>
#include <string>

using namespace std;

const int ALPHABETS = 26;

int chToIdx(char ch) { return ch - 'A'; }
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
		if (idx == s.size() - 1) {
			check = true;
			return;
		}
		int next = chToIdx(s[idx]);
		if (chil[next] == NULL)
			chil[next] = new Trie;
		chil[next]->insert(s, idx + 1);
	}
	bool find(string& s, int idx = 0) {
		if (idx == s.size() - 1)
			return check;
		int next = chToIdx(s[idx]);
		if (chil[next] == NULL)
			return false;
		chil[next]->find(s, idx + 1);
	}
};

int main(void)
{
	Trie trie;
	trie.insert(string("ASF"));

	cout << trie.find(string("DA")) << endl;
	cout << trie.find(string("AS")) << endl;
	cout << trie.find(string("ASF")) << endl;
}
