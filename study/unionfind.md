# 유니온파인드 - Union Find

유니온파인드는 Disjoint Set이라고도 불림

배열을 사용해 Tree을 구현하고 최상단 노드를 root로 하여 두 요소가 집합인지 아닌지를 구분합니다. 
root[i]는 i번째 노드의 최상단 노드를 가리키며, level[i]는 현재 그 집합(트리)의 깊이를 말합니다. 
merge는 두 집합을 합치는 기능이고 find는 최상단 노드를 찾는 함수입니다. 
시간복잡도는 merge와 find 둘다 O(logN) 입니다. 

## 주의할 점
유니온 파인드에서 서로 같은 집합인지 판별하기 위한 방법으로는 find(x)와 find(y)가 같은지 판단하면 되지만 아래 링크를 해둔 문제의 경우 level배열을 그대로 비교하면 틀리는데 끝쪽에 있는 노드들이 갱신이 안되어 있을 수 있기 때문입니다. 그래서 둘이 연결되어 있는지만 찾는 것이 아니라 전체적인 level 상태를 모두 보고 싶다면 리프노드를 모두 find처리 해주어야 갱신이 됩니다. 
https://www.acmicpc.net/problem/11724

또한 union은 cpp의 예악어이기에 merge로 변경하였으며 rank도 정확하게는 검색 안해봤으나 백준에 채점시킬 때 `reference to 'rank' is ambiguous` 라는 에러가 발생하였습니다. 그래서 level로 바꿨습니다. 

구현 단계는 다음과 같습니다. 

## init
N개의 원소가 N개의 집합에 각각 포함되도록 합니다. root[i]=i로 서로의 집합을 구분할 수 있고, level은 모두 1로 초기 깊이는 모두 1이기 때문입니다. 
```cpp
void init(int n) {
	for(int i=0; i<n; i++){
		root[i] = i;
		level[i] = 1;
	}
}
```

## merge
u와 v가 들어오면 find로 최상위 노드를 찾아주고 다음의 과정을 진행합니다. 
* find(u)와 find(y)가 같다면 바로 끝냅니다. 어짜피 이미 집합인 상태이기 때문입니다. 
* level값이 더 낮은 쪽이 큰 쪽의 자식이 됩니다. 항상 높이가 더 낮은 트리를 높은 트리 밑에 넣음으로써 LinkedList형태처럼 트리 구조가 만들어져 버리는 것을 방지합니다.
* 마지막 줄을 통해 level값을 갱신합니다. 
```cpp
void merge(int x, int y) {
	x = find(x);
	y = find(y);
	if (x == y) return;
	if (level[x] < level[y]) root[x] = y;
	else root[y] = x;
	if (level[x] == level[y]) level[x]++;
}
```

## find
주어진 원소가 속한 집합의 root를 반환합니다. 이때, 자신의 상위노드를 찾음과 동시에 최상위노드를 갱신해줌으로써 주어진 원소와 최상위 노드 사이에 있던 노드들을 따로 갱신 할 필요 없게 합니다. 
```cpp
int find(int x) {
	return root[x] == x ? x : root[x] = find(root[x]);
}
```
