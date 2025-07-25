#include "art.h"
#include <stdio.h>
#include <assert.h>
#include <queue>
#include <vector>

#if ONLINE
struct Compare
{
    bool operator()(art_node *a, art_node *b)
    {
        return a->hit_cnt > b->hit_cnt; // Min-heap
    }
};

std::priority_queue<art_node *, std::vector<art_node *>, Compare> min_heap;

extern "C" void populate_min_heap(art_node *n)
{
    if (min_heap.size() < TOP_K_SWAP)
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

// unordered set + partial_sort_top_k()
#include <unordered_set>
#include <vector>
#include <algorithm>

std::unordered_set<art_node *> node4_set_local;
std::unordered_set<art_node *> node4_set_cxl;
std::unordered_set<art_node *> node16_set_local;
std::unordered_set<art_node *> node16_set_cxl;
std::unordered_set<art_node *> node48_set_local;
std::unordered_set<art_node *> node48_set_cxl;
std::unordered_set<art_node *> node256_set_local;
std::unordered_set<art_node *> node256_set_cxl;
extern "C" void ins_node_to_set(art_node *n, uint8_t type)
{
    struct memkind *kind = memkind_detect_kind((void *)n);
    switch (type)
    {
    case NODE4:
        if (kind == node4_local_kind)
        {
            node4_set_local.insert(n);
        }
        else
            node4_set_cxl.insert(n);
        break;
    case NODE16:
        if (kind == node16_local_kind)
            node16_set_local.insert(n);
        else
            node16_set_cxl.insert(n);
        break;
    case NODE48:
        if (kind == node48_local_kind)
            node48_set_local.insert(n);
        else
            node48_set_cxl.insert(n);
        break;
    case NODE256:
        if (kind == node256_local_kind)
            node256_set_local.insert(n);
        else
            node256_set_cxl.insert(n);
        break;

    default:
        break;
    }
}
extern "C" void del_node_from_set(art_node *n, struct memkind *kind)
{
    switch (n->type)
    {
    case NODE4:
        if (kind == node4_local_kind)
            node4_set_local.erase(n);
        else
            node4_set_cxl.erase(n);
        break;
    case NODE16:
        if (kind == node16_local_kind)
            node16_set_local.erase(n);
        else
            node16_set_cxl.erase(n);
        break;
    case NODE48:
        if (kind == node48_local_kind)
            node48_set_local.erase(n);
        else
            node48_set_cxl.erase(n);
        break;
    case NODE256:
        if (kind == node256_local_kind)
            node256_set_local.erase(n);
        else
            node256_set_cxl.erase(n);
        break;

    default:
        break;
    }
}

static bool sort_by_hit_cnt_asce(const art_node *a, const art_node *b)
{
    return a->hit_cnt < b->hit_cnt; // <: ascending, >: descending
}
static bool sort_by_hit_cnt_desc(const art_node *a, const art_node *b)
{
    return a->hit_cnt > b->hit_cnt; // <: ascending, >: descending
}

static void get_top_k_nodes(art_node **out_arr, int *out_k, std::unordered_set<art_node *> &curr_set, bool ascending)
{
    std::vector<art_node *> vec(curr_set.begin(), curr_set.end());

    int actual_k = std::min(TOP_K_SWAP, static_cast<int>(vec.size()));

    if (ascending)
        std::partial_sort(vec.begin(), vec.begin() + actual_k, vec.end(), sort_by_hit_cnt_asce);
    else
        std::partial_sort(vec.begin(), vec.begin() + actual_k, vec.end(), sort_by_hit_cnt_desc);

    for (int i = 0; i < actual_k; ++i)
    {
        out_arr[i] = vec[i];
    }
    *out_k = actual_k;
}

static void get_top_k_and_swap_same_type(std::unordered_set<art_node *> &local_set, std::unordered_set<art_node *> &cxl_set)
{
    art_node *local_tmp_arr[TOP_K_SWAP];
    int local_tmp_cnt = 0;
    get_top_k_nodes(local_tmp_arr, &local_tmp_cnt, local_set, true);
    art_node *cxl_tmp_arr[TOP_K_SWAP];
    int cxl_tmp_cnt = 0;
    get_top_k_nodes(cxl_tmp_arr, &cxl_tmp_cnt, cxl_set, false);
    printf("local_tmp_cnt: %d, cxl_tmp_cnt: %d\n", local_tmp_cnt, cxl_tmp_cnt);
    // swap_hot_cold_nodes(local_tmp_arr, local_tmp_cnt, cxl_tmp_arr, cxl_tmp_cnt);
}
extern "C" void get_top_k_and_swap()
{
    get_top_k_and_swap_same_type(node4_set_local, node4_set_cxl);
    get_top_k_and_swap_same_type(node16_set_local, node16_set_cxl);
    get_top_k_and_swap_same_type(node48_set_local, node48_set_cxl);
    get_top_k_and_swap_same_type(node256_set_local, node256_set_cxl);
}
#include <memory>
struct MaxHeapCompare
{
    bool operator()(const art_node *a, const art_node *b) const
    {
        return a->hit_cnt < b->hit_cnt; // for smallest
    }
};

struct MinHeapCompare
{
    bool operator()(const art_node *a, const art_node *b) const
    {
        return a->hit_cnt > b->hit_cnt; // for largest
    }
};

extern "C" void get_top_k_by_hit_cnt(art_node **out_arr, int *out_k, art_node **arr, int N, int k, bool ascending)
{
    if (!arr || !out_arr || !out_k || N <= 0 || k <= 0)
        return;

    k = std::min(k, N);

    if (ascending)
    {
        std::priority_queue<art_node *, std::vector<art_node *>, MaxHeapCompare> p_q;

        for (int i = 0; i < N; ++i)
        {
            art_node *n = arr[i];
            if ((int)p_q.size() < k)
                p_q.push(n);
            else if (n->hit_cnt < p_q.top()->hit_cnt)
            {
                p_q.pop();
                p_q.push(n);
            }
        }

        int count = 0;
        while (!p_q.empty())
        {
            out_arr[count++] = p_q.top();
            p_q.pop();
        }

        std::sort(out_arr, out_arr + count,
                  [](const art_node *a, const art_node *b)
                  { return a->hit_cnt < b->hit_cnt; });

        *out_k = count;
    }
    else
    {
        std::priority_queue<art_node *, std::vector<art_node *>, MinHeapCompare> p_q;

        for (int i = 0; i < N; ++i)
        {
            art_node *n = arr[i];
            if ((int)p_q.size() < k)
                p_q.push(n);
            else if (n->hit_cnt > p_q.top()->hit_cnt)
            {
                p_q.pop();
                p_q.push(n);
            }
        }

        int count = 0;
        while (!p_q.empty())
        {
            out_arr[count++] = p_q.top();
            p_q.pop();
        }

        std::sort(out_arr, out_arr + count,
                  [](const art_node *a, const art_node *b)
                  { return a->hit_cnt > b->hit_cnt; });

        *out_k = count;
    }
}

#endif