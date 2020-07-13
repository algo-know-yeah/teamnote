#include <iostream>
using namespace std;

// 알고리즘 문제해결 전략

typedef int type;
struct Treap {
	Treap* left = NULL, * right = NULL;
	int size = 1, prio = rand();
	type key;
	Treap(type key) : key(key) { }
	void calc_size() {
		size = 1;
		if (left != NULL) size += left->size;
		if (right != NULL) size += right->size;
	}
	void set_left(Treap* l) { left = l, calc_size(); }
	void set_right(Treap* r) { right = r, calc_size(); }
};
typedef pair<Treap*, Treap*> TPair;
TPair split(Treap* root, type key) {
	if (root == NULL) return TPair(NULL, NULL);
	if (root->key < key) {
		TPair rs = split(root->right, key);
		root->set_right(rs.first);
		return TPair(root, rs.second);
	}
	TPair ls = split(root->left, key);
	root->set_left(ls.second);
	return TPair(ls.first, root);
}
Treap* insert(Treap* root, Treap* node) {
	if (root == NULL) return node;
	if (root->prio < node->prio) {
		TPair s = split(root, node->key);
		node->set_left(s.first);
		node->set_right(s.second);
		return node;
	}
	else if (node->key < root->key)
		root->set_left(insert(root->left, node));
	else
		root->set_right(insert(root->right, node));
	return root;
}
Treap* merge(Treap* a, Treap* b) {
	if (a == NULL) return b;
	if (b == NULL) return a;
	if (a->prio < b->prio) {
		b->set_left(merge(a, b->left));
		return b;
	}
	a->set_right(merge(a->right, b));
	return a;
}
Treap* erase(Treap* root, type key) {
	if (root == NULL) return root;
	if (root->key == key) {
		Treap* ret = merge(root->left, root->right);
		delete root;
		return ret;
	}
	if (key < root->key)
		root->set_left(erase(root->left, key));
	else
		root->set_right(erase(root->right, key));
	return root;
}
Treap* kth(Treap* root, int k) { // kth key
	int l_size = 0;
	if (root->left != NULL) l_size += root->left->size;
	if (k <= l_size) return kth(root->left, k);
	if (k == l_size + 1) return root;
	return kth(root->right, k - l_size - 1);
}
int countLess(Treap* root, type key) { // count less than key
	if (root == NULL) return 0;
	if (root->key >= key)
		return countLess(root->left, key);
	int ls = (root->left ? root->left->size : 0);
	return ls + 1 + countLess(root->right, key);
}
int main(void)
{
	Treap* root = NULL;
	root = insert(root, new Treap(3));
	root = insert(root, new Treap(6));
	root = insert(root, new Treap(2));
	root = insert(root, new Treap(7));
	root = insert(root, new Treap(1));

	cout << root->size << endl;
	for (int i = 1; i <= 5; i++)
		cout << i << "th: " << kth(root, i)->key << endl;
	cout << "countLess(6): " << countLess(root, 6);
}
