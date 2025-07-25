#include <unordered_set>
#include <unordered_map>
#include <list>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <algorithm>
#include "art.h"

#if ONLINE

static std::list<art_node *> lru_list_node4;
static std::unordered_map<art_node *, std::list<art_node *>::iterator> lru_map_node4;
static std::unordered_set<art_node *> lru_set_node4;

// too much overhead
extern "C" void hot_cache_record_access(art_node *n)
{
    switch (n->type)
    {
    case NODE4:
    {
        auto found = lru_set_node4.find(n);
        if (found != lru_set_node4.end())
        {
            // Move to front
            auto it = lru_map_node4[n];
            lru_list_node4.erase(it);
            lru_list_node4.push_front(n);
            lru_map_node4[n] = lru_list_node4.begin();
        }
        else
        {
            if ((int)lru_set_node4.size() >= HOT_CACHE_LIMIT)
            {
                // Evict least recently used
                art_node *evict = lru_list_node4.back();
                lru_list_node4.pop_back();
                lru_set_node4.erase(evict);
                lru_map_node4.erase(evict);
            }
            // Insert new node
            lru_list_node4.push_front(n);
            lru_set_node4.insert(n);
            lru_map_node4[n] = lru_list_node4.begin();
        }
        break;
    }

    default:
        break;
    }
}

extern "C" int hot_cache_snapshot(art_node **out_arr, int max_size)
{
    int count = 0;
    for (art_node *n : lru_list_node4)
    {
        if (count >= max_size)
            break;
        out_arr[count++] = n;
    }
    return count;
}

extern "C" void hot_cache_reset()
{
    lru_list_node4.clear();
    lru_map_node4.clear();
    lru_set_node4.clear();
}

// Ring buffer
static art_node *ring_buffer_node4[HOT_CACHE_LIMIT];
static art_node *ring_buffer_node16[HOT_CACHE_LIMIT];
static art_node *ring_buffer_node48[HOT_CACHE_LIMIT];
static art_node *ring_buffer_node256[HOT_CACHE_LIMIT];
static int ring_head_node4 = 0;
static int ring_head_node16 = 0;
static int ring_head_node48 = 0;
static int ring_head_node256 = 0;
static bool ring_filled_node4 = false;
static bool ring_filled_node16 = false;
static bool ring_filled_node48 = false;
static bool ring_filled_node256 = false;

extern "C" void reset_ring_buffer()
{
    int ring_head_node4 = 0;
    int ring_head_node16 = 0;
    int ring_head_node48 = 0;
    int ring_head_node256 = 0;
    bool ring_filled_node4 = false;
    bool ring_filled_node16 = false;
    bool ring_filled_node48 = false;
    bool ring_filled_node256 = false;
}

extern "C" void sample_to_ring_buffer(art_node *n)
{
    if (!n || IS_LEAF(n))
        return;
    switch (n->type)
    {
    case NODE4:
    {
        ring_buffer_node4[ring_head_node4++] = n;
        if (ring_head_node4 == HOT_CACHE_LIMIT)
        {
            ring_head_node4 = 0;
            ring_filled_node4 = true;
        }
        break;
    }
    case NODE16:
    {
        ring_buffer_node16[ring_head_node16++] = n;
        if (ring_head_node16 == HOT_CACHE_LIMIT)
        {
            ring_head_node16 = 0;
            ring_filled_node16 = true;
        }
        break;
    }
    case NODE48:
    {
        ring_buffer_node48[ring_head_node48++] = n;
        if (ring_head_node48 == HOT_CACHE_LIMIT)
        {
            ring_head_node48 = 0;
            ring_filled_node48 = true;
        }
        break;
    }
    case NODE256:
    {
        ring_buffer_node256[ring_head_node256++] = n;
        if (ring_head_node256 == HOT_CACHE_LIMIT)
        {
            ring_head_node256 = 0;
            ring_filled_node256 = true;
        }
        break;
    }

    default:
        break;
    }
}

extern "C" void get_ring_buffer_snapshot(art_node **out_arr, int *out_cnt)
{
    // int count = ring_filled ? HOT_CACHE_LIMIT : ring_head;
    // memcpy(out_arr, ring_buffer_node4, count * sizeof(art_node *));
    // *out_cnt = count;
}
#endif