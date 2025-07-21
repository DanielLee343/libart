#include "art.h"
#include <stdio.h>
#include <queue>
#include <vector>

struct Compare
{
    bool operator()(art_node *a, art_node *b)
    {
        return a->hit_cnt > b->hit_cnt; // Min-heap
    }
};

const int K = 10000;
std::priority_queue<art_node *, std::vector<art_node *>, Compare> min_heap;

extern "C" void populate_min_heap(art_node *n)
{
    if (min_heap.size() < K)
    {
        min_heap.push(n); // Still room, just insert
    }
    else if (n->hit_cnt > min_heap.top()->hit_cnt)
    {
        min_heap.pop();   // Remove current min
        min_heap.push(n); // Replace with this higher-hit node
    }
}
extern "C" void reset_min_heap()
{
    while (!min_heap.empty())
    {
        min_heap.pop();
    }
    traverse_cnt = 0;
}

extern "C" void print_min_heap_stat()
{
    printf("min_heap size: %zu, traverse_cnt: %d\n", min_heap.size(), traverse_cnt);
}