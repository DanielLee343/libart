#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <assert.h>
#include "art.h"

#ifdef __i386__
#include <emmintrin.h>
#else
#ifdef __amd64__
#include <emmintrin.h>
#endif
#endif

#define PAGE_SIZE 4096

// email
#define NUM_LEAF_PAGE 772096
#define NUM_4_PAGE 302600
#define NUM_16_PAGE 203776
#define NUM_48_PAGE 112896
#define NUM_256_PAGE 256

// randint
// #define NUM_LEAF_PAGE 2930000
// #define NUM_4_PAGE 1664000
// #define NUM_16_PAGE 480000
// #define NUM_48_PAGE 256
// #define NUM_256_PAGE 300000

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))

#ifndef DEPTH_THRESH
#define DEPTH_THRESH 50
#endif
#ifndef LEAF_CXL
#define LEAF_CXL 0
#endif
#ifndef NODE4_CXL
#define NODE4_CXL 0
#endif
#ifndef NODE16_CXL
#define NODE16_CXL 0
#endif
#ifndef NODE48_CXL
#define NODE48_CXL 0
#endif
#ifndef NODE256_CXL
#define NODE256_CXL 0
#endif

#define LOCAL_MASK 0
#define CXL_MASK 1
static const int type_map[] = {0, 4, 16, 48, 256}; // index 0 unused
/**
 * Macros to manipulate pointer tags
 */
#define IS_LEAF(x) (((uintptr_t)x & 1))
#define SET_LEAF(x) ((void *)((uintptr_t)x | 1))
#define LEAF_RAW(x) ((art_leaf *)((void *)((uintptr_t)x & ~1)))
/**
 * Allocates a node of the given type,
 * initializes to zero and sets the type.
 */
static art_node *alloc_node(uint8_t type)
{
    art_node *n;
    switch (type)
    {
    case NODE4:
// n = ALIGN_UP(node4_ptr, LEAF_ALIGN); // for prev bulk alloced region
// node4_ptr = (void *)((uintptr_t)n + sizeof(art_node4));
// n = (art_node *)calloc(1, sizeof(art_node4)); // vanilla
#if ONLINE
        n = (art_node *)tiered_calloc(&node4_local_full, node4_local_kind, node4_cxl_kind, sizeof(art_node4), node4_hot, &node4_local_alloc_cnt, node4_cold, &node4_cxl_alloc_cnt);
#else
        n = (art_node *)memkind_calloc(node4_kind, 1, sizeof(art_node4));
#endif
#if CNT
        node4_cnt++;
#endif
        break;
    case NODE16:
#if ONLINE
        n = (art_node *)tiered_calloc(&node16_local_full, node16_local_kind, node16_cxl_kind, sizeof(art_node16), node16_hot, &node16_local_alloc_cnt, node16_cold, &node16_cxl_alloc_cnt);
#else
        n = (art_node *)memkind_calloc(node16_kind, 1, sizeof(art_node16));
#endif
#if CNT
        node16_cnt++;
#endif
        break;
    case NODE48:
#if ONLINE
        n = (art_node *)tiered_calloc(&node48_local_full, node48_local_kind, node48_cxl_kind, sizeof(art_node48), node48_hot, &node48_local_alloc_cnt, node48_cold, &node48_cxl_alloc_cnt);
#else
        n = (art_node *)memkind_calloc(node48_kind, 1, sizeof(art_node48));
#endif
#if CNT
        node48_cnt++;
#endif
        break;
    case NODE256:
#if ONLINE
        n = (art_node *)memkind_calloc(node256_kind, 1, sizeof(art_node256)); // todo: fix later, currently all to local kind
                                                                              // n = (art_node *)tiered_calloc(&node256_local_kind, node256_local_kind, node256_cxl_kind, sizeof(art_node256), node256_hot, &node256_local_alloc_cnt, node256_cold, &node256_cxl_alloc_cnt);
#else
        n = (art_node *)memkind_calloc(node256_kind, 1, sizeof(art_node256));
#endif
#if CNT
        node256_cnt++;
#endif
        break;
    default:
        abort();
    }
    assert(n);
    n->type = type;
    return n;
}

/**
 * Initializes an ART tree
 * @return 0 on success.
 */
int art_tree_init(art_tree *t)
{
    SHOW_DEFINE(LEAF_CXL);
    SHOW_DEFINE(NODE4_CXL);
    SHOW_DEFINE(NODE16_CXL);
    SHOW_DEFINE(NODE48_CXL);
    SHOW_DEFINE(NODE256_CXL);
    if (STATIC_DIST)
    {
        SHOW_DEFINE(DEPTH_THRESH);
    }
    t->root = NULL;
    t->size = 0;
    init_region(&leaf_base, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, LEAF_CXL, &leaf_kind);
    init_region(&node4_base, (size_t)PAGE_SIZE * NUM_4_PAGE, NODE4_CXL, &node4_kind);
    init_region(&node16_base, (size_t)PAGE_SIZE * NUM_16_PAGE, NODE16_CXL, &node16_kind);
    init_region(&node48_base, (size_t)PAGE_SIZE * NUM_48_PAGE, NODE48_CXL, &node48_kind);
    init_region(&node256_base, (size_t)PAGE_SIZE * NUM_256_PAGE, NODE256_CXL, &node256_kind);
#if STATIC_DIST
    init_region(&leaf_local, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, 0, &leaf_local_kind);
    init_region(&leaf_cxl, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, 1, &leaf_cxl_kind);
    init_region(&node4_local, (size_t)PAGE_SIZE * NUM_4_PAGE, 0, &node4_local_kind);
    init_region(&node4_cxl, (size_t)PAGE_SIZE * NUM_4_PAGE, 1, &node4_cxl_kind);
    init_region(&node16_local, (size_t)PAGE_SIZE * NUM_16_PAGE, 0, &node16_local_kind);
    init_region(&node16_cxl, (size_t)PAGE_SIZE * NUM_16_PAGE, 1, &node16_cxl_kind);
    init_region(&node48_local, (size_t)PAGE_SIZE * NUM_48_PAGE, 0, &node48_local_kind);
    init_region(&node48_cxl, (size_t)PAGE_SIZE * NUM_48_PAGE, 1, &node48_cxl_kind);
    init_region(&node256_local, (size_t)PAGE_SIZE * NUM_256_PAGE, 0, &node256_local_kind);
    // init_region(&node256_cxl, (size_t)PAGE_SIZE * NUM_256_PAGE, 1, &node256_cxl_kind); // do not use due to jemalloc hard-limits
#elif ONLINE
    init_region(&leaf_local, (size_t)PAGE_SIZE * NUM_LEAF_PAGE / 2, 0, &leaf_local_kind);
    init_region(&leaf_cxl, (size_t)PAGE_SIZE * NUM_LEAF_PAGE / 2, 1, &leaf_cxl_kind);
    init_region(&node4_local, (size_t)PAGE_SIZE * NUM_4_PAGE / 2, 0, &node4_local_kind);
    init_region(&node4_cxl, (size_t)PAGE_SIZE * NUM_4_PAGE / 2, 1, &node4_cxl_kind);
    init_region(&node16_local, (size_t)PAGE_SIZE * NUM_16_PAGE / 2, 0, &node16_local_kind);
    init_region(&node16_cxl, (size_t)PAGE_SIZE * NUM_16_PAGE / 2, 1, &node16_cxl_kind);
    init_region(&node48_local, (size_t)PAGE_SIZE * NUM_48_PAGE / 2, 0, &node48_local_kind);
    init_region(&node48_cxl, (size_t)PAGE_SIZE * NUM_48_PAGE / 2, 1, &node48_cxl_kind);
    init_region(&node256_local, (size_t)PAGE_SIZE * NUM_256_PAGE / 2, 0, &node256_local_kind);
    // init_region(&node256_cxl, (size_t)PAGE_SIZE * NUM_256_PAGE / 2, 1, &node256_cxl_kind); // do not use due to jemalloc hard-limits
    node4_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_4_PAGE / 2) / sizeof(art_node4)) * sizeof(void *));
    node4_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_4_PAGE / 2) / sizeof(art_node4)) * sizeof(void *));
    node16_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_16_PAGE / 2) / sizeof(art_node16)) * sizeof(void *));
    node16_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_16_PAGE / 2) / sizeof(art_node16)) * sizeof(void *));
    node48_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_256_PAGE / 2) / sizeof(art_node256)) * sizeof(void *));
    node48_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_256_PAGE / 2) / sizeof(art_node256)) * sizeof(void *));
#endif
    return 0;
}

// Recursively destroys the tree
static void destroy_node(art_node *n)
{
    // Break if null
    if (!n)
        return;

    // Special case leafs
    if (IS_LEAF(n))
    {
#if CNT
        leaf_cnt--;
#endif
        struct memkind *kind = memkind_detect_kind((void *)LEAF_RAW(n));
        memkind_free(kind, n);
        return;
    }

    // Handle each node type
    int i, idx;
    union
    {
        art_node4 *p1;
        art_node16 *p2;
        art_node48 *p3;
        art_node256 *p4;
    } p;
    struct memkind *kind = memkind_detect_kind((void *)n);
    switch (n->type)
    {
    case NODE4:
        p.p1 = (art_node4 *)n;
        for (i = 0; i < n->num_children; i++)
        {
            destroy_node(p.p1->children[i]);
        }
        memkind_free(kind, n);
#if CNT
        node4_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr(n, &node4_local_alloc_cnt, &node4_cxl_alloc_cnt, node4_hot, node4_cold);
#endif
        break;

    case NODE16:
        p.p2 = (art_node16 *)n;
        for (i = 0; i < n->num_children; i++)
        {
            destroy_node(p.p2->children[i]);
        }
        memkind_free(kind, n);
#if CNT
        node16_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr(n, &node16_local_alloc_cnt, &node16_cxl_alloc_cnt, node16_hot, node16_cold);
#endif
        break;

    case NODE48:
        p.p3 = (art_node48 *)n;
        for (i = 0; i < 256; i++)
        {
            idx = ((art_node48 *)n)->keys[i];
            if (!idx)
                continue;
            destroy_node(p.p3->children[idx - 1]);
        }
        memkind_free(kind, n);
#if CNT
        node48_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr(n, &node48_local_alloc_cnt, &node48_cxl_alloc_cnt, node48_hot, node48_cold);
#endif
        break;

    case NODE256:
        p.p4 = (art_node256 *)n;
        for (i = 0; i < 256; i++)
        {
            if (p.p4->children[i])
                destroy_node(p.p4->children[i]);
        }
        memkind_free(kind, n);
#if CNT
        node256_cnt--;
#endif
        break;

    default:
        abort();
    }

    // Free ourself on the way up
    // printf("%d\n", n->type);
    // free(n);
}

/**
 * Destroys an ART tree
 * @return 0 on success.
 */
int art_tree_destroy(art_tree *t)
{
    destroy_node(t->root);
    // numa_free(slab_base, (size_t)PAGE_SIZE * NUM_LEAF_PAGE); // for prev bulk free
    destroy_region(leaf_base, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, leaf_kind);
    destroy_region(node4_base, (size_t)PAGE_SIZE * NUM_4_PAGE, node4_kind);
    destroy_region(node16_base, (size_t)PAGE_SIZE * NUM_16_PAGE, node16_kind);
    destroy_region(node48_base, (size_t)PAGE_SIZE * NUM_48_PAGE, node48_kind);
    destroy_region(node256_base, (size_t)PAGE_SIZE * NUM_256_PAGE, node256_kind);
#if STATIC_DIST
    destroy_region(leaf_local, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, leaf_local_kind);
    destroy_region(leaf_cxl, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, leaf_cxl_kind);
    destroy_region(node4_local, (size_t)PAGE_SIZE * NUM_4_PAGE, node4_local_kind);
    destroy_region(node4_cxl, (size_t)PAGE_SIZE * NUM_4_PAGE, node4_cxl_kind);
    destroy_region(node16_local, (size_t)PAGE_SIZE * NUM_16_PAGE, node16_local_kind);
    destroy_region(node16_cxl, (size_t)PAGE_SIZE * NUM_16_PAGE, node16_cxl_kind);
    destroy_region(node48_local, (size_t)PAGE_SIZE * NUM_48_PAGE, node48_local_kind);
    destroy_region(node48_cxl, (size_t)PAGE_SIZE * NUM_48_PAGE, node48_cxl_kind);
    destroy_region(node256_local, (size_t)PAGE_SIZE * NUM_256_PAGE, node256_local_kind);
    // destroy_region(node256_cxl, (size_t)PAGE_SIZE * NUM_256_PAGE, node256_cxl_kind);
#elif ONLINE
    destroy_region(leaf_local, (size_t)PAGE_SIZE * NUM_LEAF_PAGE / 2, leaf_local_kind);
    destroy_region(leaf_cxl, (size_t)PAGE_SIZE * NUM_LEAF_PAGE / 2, leaf_cxl_kind);
    destroy_region(node4_local, (size_t)PAGE_SIZE * NUM_4_PAGE / 2, node4_local_kind);
    destroy_region(node4_cxl, (size_t)PAGE_SIZE * NUM_4_PAGE / 2, node4_cxl_kind);
    destroy_region(node16_local, (size_t)PAGE_SIZE * NUM_16_PAGE / 2, node16_local_kind);
    destroy_region(node16_cxl, (size_t)PAGE_SIZE * NUM_16_PAGE / 2, node16_cxl_kind);
    destroy_region(node48_local, (size_t)PAGE_SIZE * NUM_48_PAGE / 2, node48_local_kind);
    destroy_region(node48_cxl, (size_t)PAGE_SIZE * NUM_48_PAGE / 2, node48_cxl_kind);
    destroy_region(node256_local, (size_t)PAGE_SIZE * NUM_256_PAGE / 2, node256_local_kind);
    // destroy_region(node256_cxl, (size_t)PAGE_SIZE * NUM_256_PAGE / 2, node256_cxl_kind);
    free(node4_hot);
    free(node4_cold);
    free(node16_hot);
    free(node16_cold);
    free(node48_hot);
    free(node48_cold);
#endif
    return 0;
}

/**
 * Returns the size of the ART tree.
 */

#ifndef BROKEN_GCC_C99_INLINE
extern inline uint64_t art_size(art_tree *t);
#endif

static art_node **find_child(art_node *n, unsigned char c)
{
    int i, mask, bitfield;
    union
    {
        art_node4 *p1;
        art_node16 *p2;
        art_node48 *p3;
        art_node256 *p4;
    } p;
    switch (n->type)
    {
    case NODE4:
        p.p1 = (art_node4 *)n;
        for (i = 0; i < n->num_children; i++)
        {
            /* this cast works around a bug in gcc 5.1 when unrolling loops
             * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59124
             */
            if (((unsigned char *)p.p1->keys)[i] == c)
                return &p.p1->children[i];
        }
        break;

        {
        case NODE16:
            p.p2 = (art_node16 *)n;

// support non-86 architectures
#ifdef __i386__
            // Compare the key to all 16 stored keys
            __m128i cmp;
            cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                                 _mm_loadu_si128((__m128i *)p.p2->keys));

            // Use a mask to ignore children that don't exist
            mask = (1 << n->num_children) - 1;
            bitfield = _mm_movemask_epi8(cmp) & mask;
#else
#ifdef __amd64__
            // Compare the key to all 16 stored keys
            __m128i cmp;
            cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                                 _mm_loadu_si128((__m128i *)p.p2->keys));

            // Use a mask to ignore children that don't exist
            mask = (1 << n->num_children) - 1;
            bitfield = _mm_movemask_epi8(cmp) & mask;
#else
            // Compare the key to all 16 stored keys
            bitfield = 0;
            for (i = 0; i < 16; ++i)
            {
                if (p.p2->keys[i] == c)
                    bitfield |= (1 << i);
            }

            // Use a mask to ignore children that don't exist
            mask = (1 << n->num_children) - 1;
            bitfield &= mask;
#endif
#endif

            /*
             * If we have a match (any bit set) then we can
             * return the pointer match using ctz to get
             * the index.
             */
            if (bitfield)
                return &p.p2->children[__builtin_ctz(bitfield)];
            break;
        }

    case NODE48:
        p.p3 = (art_node48 *)n;
        i = p.p3->keys[c];
        if (i)
            return &p.p3->children[i - 1];
        break;

    case NODE256:
        p.p4 = (art_node256 *)n;
        if (p.p4->children[c])
            return &p.p4->children[c];
        break;

    default:
        abort();
    }
    return NULL;
}

// Simple inlined if
static inline int min(int a, int b)
{
    return (a < b) ? a : b;
}

/**
 * Returns the number of prefix characters shared between
 * the key and node.
 */
static int check_prefix(const art_node *n, const unsigned char *key, int key_len, int depth)
{
    int max_cmp = min(min(n->partial_len, MAX_PREFIX_LEN), key_len - depth);
    int idx;
    for (idx = 0; idx < max_cmp; idx++)
    {
        if (n->partial[idx] != key[depth + idx])
            return idx;
    }
    return idx;
}

/**
 * Checks if a leaf matches
 * @return 0 on success.
 */
static int leaf_matches(const art_leaf *n, const unsigned char *key, int key_len, int depth)
{
    (void)depth;
    // Fail if the key lengths are different
    if (n->key_len != (uint32_t)key_len)
        return 1;

    // Compare the keys starting at the depth
    return memcmp(n->key, key, key_len);
}

/**
 * Searches for a value in the ART tree
 * @arg t The tree
 * @arg key The key
 * @arg key_len The length of the key
 * @return NULL if the item was not found, otherwise
 * the value pointer is returned.
 */
void *art_search(const art_tree *t, const unsigned char *key, int key_len)
{
    art_node **child;
    art_node *n = t->root;
    int prefix_len, depth = 0;
    while (n)
    {
        // Might be a leaf
        if (IS_LEAF(n))
        {
#if HIT_CNT_TOTAL
            leaf_hit_cnt++;
#endif
#if STREAM_ACC_ADDR
            if (start_acc_streaming)
                fprintf(acc_fd, "%lu,0\n", (unsigned long)n);
#endif
            n = (art_node *)LEAF_RAW(n);
            // Check if the expanded path matches
            if (!leaf_matches((art_leaf *)n, key, key_len, depth))
            {
                return ((art_leaf *)n)->value;
            }
            return NULL;
        }
#if HIT_CNT_TOTAL
        switch (n->type)
        {
        case NODE4:
            node4_hit_cnt++;
            break;
        case NODE16:
            node16_hit_cnt++;
            break;
        case NODE48:
            node48_hit_cnt++;
            break;
        case NODE256:
            node256_hit_cnt++;
            break;
        }
#endif
#if HIT_DIST
        n->hit_cnt++;
#endif
#if STREAM_ACC_ADDR
        if (start_acc_streaming)
            fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
#endif

        // Bail if the prefix does not match
        if (n->partial_len)
        {
            prefix_len = check_prefix(n, key, key_len, depth);
            if (prefix_len != min(MAX_PREFIX_LEN, n->partial_len))
                return NULL;
            depth = depth + n->partial_len;
        }

        // Recursively search
        child = find_child(n, key[depth]);
        n = (child) ? *child : NULL;
        depth++;
    }
    return NULL;
}

// Find the minimum leaf under a node
static art_leaf *minimum(const art_node *n)
{
    // Handle base cases
    if (!n)
        return NULL;
    if (IS_LEAF(n))
        return LEAF_RAW(n);

    int idx;
    switch (n->type)
    {
    case NODE4:
        return minimum(((const art_node4 *)n)->children[0]);
    case NODE16:
        return minimum(((const art_node16 *)n)->children[0]);
    case NODE48:
        idx = 0;
        while (!((const art_node48 *)n)->keys[idx])
            idx++;
        idx = ((const art_node48 *)n)->keys[idx] - 1;
        return minimum(((const art_node48 *)n)->children[idx]);
    case NODE256:
        idx = 0;
        while (!((const art_node256 *)n)->children[idx])
            idx++;
        return minimum(((const art_node256 *)n)->children[idx]);
    default:
        abort();
    }
}

// Find the maximum leaf under a node
static art_leaf *maximum(const art_node *n)
{
    // Handle base cases
    if (!n)
        return NULL;
    if (IS_LEAF(n))
        return LEAF_RAW(n);

    int idx;
    switch (n->type)
    {
    case NODE4:
        return maximum(((const art_node4 *)n)->children[n->num_children - 1]);
    case NODE16:
        return maximum(((const art_node16 *)n)->children[n->num_children - 1]);
    case NODE48:
        idx = 255;
        while (!((const art_node48 *)n)->keys[idx])
            idx--;
        idx = ((const art_node48 *)n)->keys[idx] - 1;
        return maximum(((const art_node48 *)n)->children[idx]);
    case NODE256:
        idx = 255;
        while (!((const art_node256 *)n)->children[idx])
            idx--;
        return maximum(((const art_node256 *)n)->children[idx]);
    default:
        abort();
    }
}

/**
 * Returns the minimum valued leaf
 */
art_leaf *art_minimum(art_tree *t)
{
    return minimum((art_node *)t->root);
}

/**
 * Returns the maximum valued leaf
 */
art_leaf *art_maximum(art_tree *t)
{
    return maximum((art_node *)t->root);
}

static art_leaf *make_leaf(const unsigned char *key, int key_len, void *value)
{
    art_leaf *l = NULL;
// l = ALIGN_UP(bump_ptr, LEAF_ALIGN); // bulk alloc version
// bump_ptr = (void *)((uintptr_t)l + sizeof(art_leaf) + key_len);
// l = (art_leaf *)calloc(1, sizeof(art_leaf) + key_len); // vanilla
#if ONLINE
    // if (!leaf_local_full)
    // {
    //     l = memkind_calloc(leaf_local_kind, 1, sizeof(art_leaf) + key_len);
    // }
    // if (!l)
    // {
    //     leaf_local_full = 1;
    //     l = memkind_calloc(leaf_cxl_kind, 1, sizeof(art_leaf) + key_len);
    // }
    // l = (art_leaf *)tiered_calloc(&leaf_local_full, leaf_local_kind, leaf_cxl_kind, sizeof(art_leaf) + key_len);
    l = (art_leaf *)memkind_calloc(leaf_kind, 1, sizeof(art_leaf) + key_len); // currently we don't consider leaf movement
#else
    l = (art_leaf *)memkind_calloc(leaf_kind, 1, sizeof(art_leaf) + key_len); // compare addr pattern
#endif
    // printf("0x%lx, size: %lu\n", (unsigned long)(uintptr_t)l, sizeof(art_leaf) + key_len);
    l->value = value;
    l->key_len = key_len;
    memcpy(l->key, key, key_len);
#if CNT
    leaf_cnt++;
#endif
    return l;
}

static int longest_common_prefix(art_leaf *l1, art_leaf *l2, int depth)
{
    int max_cmp = min(l1->key_len, l2->key_len) - depth;
    int idx;
    for (idx = 0; idx < max_cmp; idx++)
    {
        if (l1->key[depth + idx] != l2->key[depth + idx])
            return idx;
    }
    return idx;
}

static void copy_header(art_node *dest, art_node *src)
{
    dest->num_children = src->num_children;
    dest->partial_len = src->partial_len;
    memcpy(dest->partial, src->partial, min(MAX_PREFIX_LEN, src->partial_len));
}

static void add_child256(art_node256 *n, art_node **ref, unsigned char c, void *child)
{
    (void)ref;
    n->n.num_children++;
    n->children[c] = (art_node *)child;
}

static void add_child48(art_node48 *n, art_node **ref, unsigned char c, void *child)
{
    art_node *child_node = (art_node *)child;
    if (n->n.num_children < 48)
    {
        int pos = 0;
        while (n->children[pos])
            pos++;
        n->children[pos] = (art_node *)child;
        n->keys[c] = pos + 1;
        n->n.num_children++;
    }
    else
    {
        art_node256 *new_node = (art_node256 *)alloc_node(NODE256);

        for (int i = 0; i < 256; i++)
        {
            if (n->keys[i])
            {
                art_node *existing_child = n->children[n->keys[i] - 1];
                new_node->children[i] = existing_child;
            }
        }
        copy_header((art_node *)new_node, (art_node *)n);
        *ref = (art_node *)new_node;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if CNT
        node48_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr((art_node *)n, &node48_local_alloc_cnt, &node48_cxl_alloc_cnt, node48_hot, node48_cold);
#endif
        add_child256(new_node, ref, c, child);
    }
}

static void add_child16(art_node16 *n, art_node **ref, unsigned char c, void *child)
{
    art_node *child_node = (art_node *)child;

    if (n->n.num_children < 16)
    {
        unsigned mask = (1 << n->n.num_children) - 1;

#ifdef __i386__
        __m128i cmp = _mm_cmplt_epi8(_mm_set1_epi8(c), _mm_loadu_si128((__m128i *)n->keys));
        unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#elif defined(__amd64__)
        __m128i cmp = _mm_cmplt_epi8(_mm_set1_epi8(c), _mm_loadu_si128((__m128i *)n->keys));
        unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
        unsigned bitfield = 0;
        for (short i = 0; i < 16; ++i)
        {
            if (c < n->keys[i])
                bitfield |= (1 << i);
        }
        bitfield &= mask;
#endif

        unsigned idx;
        if (bitfield)
        {
            idx = __builtin_ctz(bitfield);
            memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
            memmove(n->children + idx + 1, n->children + idx,
                    (n->n.num_children - idx) * sizeof(void *));
        }
        else
        {
            idx = n->n.num_children;
        }

        n->keys[idx] = c;
        n->children[idx] = child_node;
        n->n.num_children++;
    }
    else
    {
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48);

        // Copy existing children
        for (int i = 0; i < n->n.num_children; i++)
        {
            unsigned char k = n->keys[i];
            new_node->keys[k] = i + 1;
            new_node->children[i] = n->children[i];
        }

        copy_header((art_node *)new_node, (art_node *)n);
        *ref = (art_node *)new_node;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);

#if CNT
        node16_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr((art_node *)n, &node16_local_alloc_cnt, &node16_cxl_alloc_cnt, node16_hot, node16_cold);
#endif
        add_child48(new_node, ref, c, child);
    }
}

static void add_child4(art_node4 *n, art_node **ref, unsigned char c, void *child)
{
    art_node *child_node = (art_node *)child;

    if (n->n.num_children < 4)
    {
        int idx;
        for (idx = 0; idx < n->n.num_children; idx++)
        {
            if (c < n->keys[idx])
                break;
        }

        // Shift to make room
        memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
        memmove(n->children + idx + 1, n->children + idx,
                (n->n.num_children - idx) * sizeof(void *));

        // Insert element
        n->keys[idx] = c;
        n->children[idx] = child_node;
        n->n.num_children++;
    }
    else
    {
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16);

        // Copy the child pointers and the key map
        memcpy(new_node->children, n->children,
               sizeof(void *) * n->n.num_children);
        memcpy(new_node->keys, n->keys,
               sizeof(unsigned char) * n->n.num_children);

        copy_header((art_node *)new_node, (art_node *)n);
        *ref = (art_node *)new_node;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if CNT
        node4_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr((art_node *)n, &node4_local_alloc_cnt, &node4_cxl_alloc_cnt, node4_hot, node4_cold);
#endif
        add_child16(new_node, ref, c, child);
    }
}

static void add_child(art_node *n, art_node **ref, unsigned char c, void *child)
{
    switch (n->type)
    {
    case NODE4:
        return add_child4((art_node4 *)n, ref, c, child);
    case NODE16:
        return add_child16((art_node16 *)n, ref, c, child);
    case NODE48:
        return add_child48((art_node48 *)n, ref, c, child);
    case NODE256:
        return add_child256((art_node256 *)n, ref, c, child);
    default:
        abort();
    }
}

/**
 * Calculates the index at which the prefixes mismatch
 */
static int prefix_mismatch(const art_node *n, const unsigned char *key, int key_len, int depth)
{
    int max_cmp = min(min(MAX_PREFIX_LEN, n->partial_len), key_len - depth);
    int idx;
    for (idx = 0; idx < max_cmp; idx++)
    {
        if (n->partial[idx] != key[depth + idx])
            return idx;
    }

    // If the prefix is short we can avoid finding a leaf
    if (n->partial_len > MAX_PREFIX_LEN)
    {
        // Prefix is longer than what we've checked, find a leaf
        art_leaf *l = minimum(n);
        max_cmp = min(l->key_len, key_len) - depth;
        for (; idx < max_cmp; idx++)
        {
            if (l->key[idx + depth] != key[depth + idx])
                return idx;
        }
    }
    return idx;
}

static void *recursive_insert(art_node *n, art_node **ref, const unsigned char *key, int key_len, void *value, int depth, int *old, int replace)
{
    // If we are at a NULL node, inject a leaf
    if (!n)
    {
        *ref = (art_node *)SET_LEAF(make_leaf(key, key_len, value));
        return NULL;
    }

    // If we are at a leaf, we need to replace it with a node
    if (IS_LEAF(n))
    {
#if HIT_CNT_TOTAL
        leaf_hit_cnt++;
#endif
#if STREAM_ACC_ADDR
        if (start_acc_streaming)
            fprintf(acc_fd, "%lu,0\n", (unsigned long)n);
#endif
        // printf("n = %p\n", n);
        art_leaf *l = LEAF_RAW(n);
        // printf("l = %p\n", l);
        // printf("  l->key_len = %u\n", l->key_len); // may crash here
        // Check if we are updating an existing value
        if (!leaf_matches(l, key, key_len, depth))
        {
            *old = 1;
            void *old_val = l->value;
            if (replace)
                l->value = value;
            return old_val;
        }

        // New value, we must split the leaf into a node4
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);

        // Create a new leaf
        art_leaf *l2 = make_leaf(key, key_len, value);

        // Determine longest prefix
        int longest_prefix = longest_common_prefix(l, l2, depth);
        new_node->n.partial_len = longest_prefix;
        memcpy(new_node->n.partial, key + depth, min(MAX_PREFIX_LEN, longest_prefix));
        // Add the leafs to the new node4
        *ref = (art_node *)new_node;
        add_child4(new_node, ref, l->key[depth + longest_prefix], SET_LEAF(l));
        add_child4(new_node, ref, l2->key[depth + longest_prefix], SET_LEAF(l2));
        return NULL;
    }
#if HIT_CNT_TOTAL
    switch (n->type)
    {
    case NODE4:
        node4_hit_cnt++;
        break;
    case NODE16:
        node16_hit_cnt++;
        break;
    case NODE48:
        node48_hit_cnt++;
        break;
    case NODE256:
        node256_hit_cnt++;
        break;
    }
#endif
#if HIT_DIST
    n->hit_cnt++;
#endif
#if STREAM_ACC_ADDR
    if (start_acc_streaming)
        fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
#endif

    // Check if given node has a prefix
    if (n->partial_len)
    {
        // Determine if the prefixes differ, since we need to split
        int prefix_diff = prefix_mismatch(n, key, key_len, depth);
        if ((uint32_t)prefix_diff >= n->partial_len)
        {
            depth += n->partial_len;
            goto RECURSE_SEARCH;
        }

        // Create a new node
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
        *ref = (art_node *)new_node;
        new_node->n.partial_len = prefix_diff;
        memcpy(new_node->n.partial, n->partial, min(MAX_PREFIX_LEN, prefix_diff));

        // Adjust the prefix of the old node
        if (n->partial_len <= MAX_PREFIX_LEN)
        {
            add_child4(new_node, ref, n->partial[prefix_diff], n);
            n->partial_len -= (prefix_diff + 1);
            memmove(n->partial, n->partial + prefix_diff + 1,
                    min(MAX_PREFIX_LEN, n->partial_len));
        }
        else
        {
            n->partial_len -= (prefix_diff + 1);
            art_leaf *l = minimum(n);
            add_child4(new_node, ref, l->key[depth + prefix_diff], n);
            memcpy(n->partial, l->key + depth + prefix_diff + 1,
                   min(MAX_PREFIX_LEN, n->partial_len));
        }

        // Insert the new leaf
        art_leaf *l = make_leaf(key, key_len, value);
        add_child4(new_node, ref, key[depth + prefix_diff], SET_LEAF(l));
        return NULL;
    }

RECURSE_SEARCH:;

    // Find a child to recurse to
    art_node **child = find_child(n, key[depth]);
    if (child)
    {
        return recursive_insert(*child, child, key, key_len, value, depth + 1, old, replace);
    }

    // No child, node goes within us
    art_leaf *l = make_leaf(key, key_len, value);
    add_child(n, ref, key[depth], SET_LEAF(l));
    return NULL;
}

/**
 * inserts a new value into the art tree
 * @arg t the tree
 * @arg key the key
 * @arg key_len the length of the key
 * @arg value opaque value.
 * @return null if the item was newly inserted, otherwise
 * the old value pointer is returned.
 */
void *art_insert(art_tree *t, const unsigned char *key, int key_len, void *value)
{
    int old_val = 0;
    void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0, &old_val, 1);
    if (!old_val)
        t->size++;
    return old;
}

/**
 * inserts a new value into the art tree (no replace)
 * @arg t the tree
 * @arg key the key
 * @arg key_len the length of the key
 * @arg value opaque value.
 * @return null if the item was newly inserted, otherwise
 * the old value pointer is returned.
 */
void *art_insert_no_replace(art_tree *t, const unsigned char *key, int key_len, void *value)
{
    int old_val = 0;
    void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0, &old_val, 0);
    if (!old_val)
        t->size++;
    return old;
}

static void remove_child256(art_node256 *n, art_node **ref, unsigned char c)
{
    n->children[c] = NULL;
    n->n.num_children--;

    // Resize to a node48 on underflow
    if (n->n.num_children == 37)
    {
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48);

        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n);

        int pos = 0;
        for (int i = 0; i < 256; i++)
        {
            if (n->children[i])
            {
                new_node->children[pos] = n->children[i];
                new_node->keys[i] = pos + 1;

                pos++;
            }
        }
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if CNT
        node256_cnt--;
#endif
    }
}

static void remove_child48(art_node48 *n, art_node **ref, unsigned char c)
{
    int pos = n->keys[c];
    n->keys[c] = 0;
    n->children[pos - 1] = NULL;
    n->n.num_children--;

    if (n->n.num_children == 12)
    {
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16);

        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n);

        int child = 0;
        for (int i = 0; i < 256; i++)
        {
            pos = n->keys[i];
            if (pos)
            {
                new_node->keys[child] = i;
                new_node->children[child] = n->children[pos - 1];

                child++;
            }
        }
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if CNT
        node48_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr((art_node *)n, &node48_local_alloc_cnt, &node48_cxl_alloc_cnt, node48_hot, node48_cold);
#endif
    }
}

static void remove_child16(art_node16 *n, art_node **ref, art_node **l)
{
    int pos = l - n->children;

    // Shift keys and children left to fill the removed slot
    memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - 1 - pos);
    memmove(n->children + pos, n->children + pos + 1,
            (n->n.num_children - 1 - pos) * sizeof(void *));
    n->n.num_children--;

    // Downgrade to node4 if number of children drops to 3
    if (n->n.num_children == 3)
    {
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);

        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n);

        memcpy(new_node->keys, n->keys, 3); // only 3 keys remain
        memcpy(new_node->children, n->children, 3 * sizeof(void *));
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if CNT
        node16_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr((art_node *)n, &node16_local_alloc_cnt, &node16_cxl_alloc_cnt, node16_hot, node16_cold);
#endif
    }
}

static void remove_child4(art_node4 *n, art_node **ref, art_node **l)
{
    int pos = l - n->children;
    memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - 1 - pos);
    memmove(n->children + pos, n->children + pos + 1, (n->n.num_children - 1 - pos) * sizeof(void *));
    n->n.num_children--;

    // Remove nodes with only a single child
    if (n->n.num_children == 1)
    {
        art_node *child = n->children[0];
        if (!IS_LEAF(child))
        {
            // Concatenate the prefixes
            int prefix = n->n.partial_len;
            if (prefix < MAX_PREFIX_LEN)
            {
                n->n.partial[prefix] = n->keys[0];
                prefix++;
            }
            if (prefix < MAX_PREFIX_LEN)
            {
                int sub_prefix = min(child->partial_len, MAX_PREFIX_LEN - prefix);
                memcpy(n->n.partial + prefix, child->partial, sub_prefix);
                prefix += sub_prefix;
            }

            // Store the prefix in the child
            memcpy(child->partial, n->n.partial, min(prefix, MAX_PREFIX_LEN));
            child->partial_len += n->n.partial_len + 1;
        }
        *ref = child;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if CNT
        node4_cnt--;
#endif
#if ONLINE
        update_hot_cold_arr((art_node *)n, &node4_local_alloc_cnt, &node4_cxl_alloc_cnt, node4_hot, node4_cold);
#endif
    }
}

static void remove_child(art_node *n, art_node **ref, unsigned char c, art_node **l)
{
    switch (n->type)
    {
    case NODE4:
        return remove_child4((art_node4 *)n, ref, l);
    case NODE16:
        return remove_child16((art_node16 *)n, ref, l);
    case NODE48:
        return remove_child48((art_node48 *)n, ref, c);
    case NODE256:
        return remove_child256((art_node256 *)n, ref, c);
    default:
        abort();
    }
}

static art_leaf *recursive_delete(art_node *n, art_node **ref, const unsigned char *key, int key_len, int depth)
{
    // Search terminated
    if (!n)
        return NULL;

    // Handle hitting a leaf node
    if (IS_LEAF(n))
    {
#if HIT_CNT_TOTAL
        leaf_hit_cnt++;
#endif
#if STREAM_ACC_ADDR
        if (start_acc_streaming)
            fprintf(acc_fd, "%lu,0\n", (unsigned long)n);
#endif
        art_leaf *l = LEAF_RAW(n);
        if (!leaf_matches(l, key, key_len, depth))
        {
            *ref = NULL;
            return l;
        }
        return NULL;
    }
#if HIT_CNT_TOTAL
    switch (n->type)
    {
    case NODE4:
        node4_hit_cnt++;
        break;
    case NODE16:
        node16_hit_cnt++;
        break;
    case NODE48:
        node48_hit_cnt++;
        break;
    case NODE256:
        node256_hit_cnt++;
        break;
    }
#endif
#if HIT_DIST
    n->hit_cnt++;
#endif
#if STREAM_ACC_ADDR
    if (start_acc_streaming)
        fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
#endif

    // Bail if the prefix does not match
    if (n->partial_len)
    {
        int prefix_len = check_prefix(n, key, key_len, depth);
        if (prefix_len != min(MAX_PREFIX_LEN, n->partial_len))
        {
            return NULL;
        }
        depth = depth + n->partial_len;
    }

    // Find child node
    art_node **child = find_child(n, key[depth]);
    if (!child)
        return NULL;

    // If the child is leaf, delete from this node
    if (IS_LEAF(*child))
    {
        art_leaf *l = LEAF_RAW(*child);
        if (!leaf_matches(l, key, key_len, depth))
        {
            remove_child(n, ref, key[depth], child);
            return l;
        }
        return NULL;

        // Recurse
    }
    else
    {
        return recursive_delete(*child, child, key, key_len, depth + 1);
    }
}

/**
 * Deletes a value from the ART tree
 * @arg t The tree
 * @arg key The key
 * @arg key_len The length of the key
 * @return NULL if the item was not found, otherwise
 * the value pointer is returned.
 */
void *art_delete(art_tree *t, const unsigned char *key, int key_len)
{
    art_leaf *l = recursive_delete(t->root, &t->root, key, key_len, 0);
    if (l)
    {
        t->size--;
        void *old = l->value;
#if CNT
        leaf_cnt--;
#endif
        // free(l);
        struct memkind *kind = memkind_detect_kind((void *)l);
        memkind_free(kind, l);
        return old;
    }
    return NULL;
}

// Recursively iterates over the tree
static int recursive_iter(art_node *n, art_callback cb, void *data)
{
    // Handle base cases
    if (!n)
        return 0;
    if (IS_LEAF(n))
    {
        art_leaf *l = LEAF_RAW(n);
        return cb(data, (const unsigned char *)l->key, l->key_len, l->value);
    }

    int idx, res;
    switch (n->type)
    {
    case NODE4:
        for (int i = 0; i < n->num_children; i++)
        {
            res = recursive_iter(((art_node4 *)n)->children[i], cb, data);
            if (res)
                return res;
        }
        break;

    case NODE16:
        for (int i = 0; i < n->num_children; i++)
        {
            res = recursive_iter(((art_node16 *)n)->children[i], cb, data);
            if (res)
                return res;
        }
        break;

    case NODE48:
        for (int i = 0; i < 256; i++)
        {
            idx = ((art_node48 *)n)->keys[i];
            if (!idx)
                continue;

            res = recursive_iter(((art_node48 *)n)->children[idx - 1], cb, data);
            if (res)
                return res;
        }
        break;

    case NODE256:
        for (int i = 0; i < 256; i++)
        {
            if (!((art_node256 *)n)->children[i])
                continue;
            res = recursive_iter(((art_node256 *)n)->children[i], cb, data);
            if (res)
                return res;
        }
        break;

    default:
        abort();
    }
    return 0;
}

/**
 * Iterates through the entries pairs in the map,
 * invoking a callback for each. The call back gets a
 * key, value for each and returns an integer stop value.
 * If the callback returns non-zero, then the iteration stops.
 * @arg t The tree to iterate over
 * @arg cb The callback function to invoke
 * @arg data Opaque handle passed to the callback
 * @return 0 on success, or the return of the callback.
 */
int art_iter(art_tree *t, art_callback cb, void *data)
{
    return recursive_iter(t->root, cb, data);
}

/**
 * Checks if a leaf prefix matches
 * @return 0 on success.
 */
static int leaf_prefix_matches(const art_leaf *n, const unsigned char *prefix, int prefix_len)
{
    // Fail if the key length is too short
    if (n->key_len < (uint32_t)prefix_len)
        return 1;

    // Compare the keys
    return memcmp(n->key, prefix, prefix_len);
}

/**
 * Iterates through the entries pairs in the map,
 * invoking a callback for each that matches a given prefix.
 * The call back gets a key, value for each and returns an integer stop value.
 * If the callback returns non-zero, then the iteration stops.
 * @arg t The tree to iterate over
 * @arg prefix The prefix of keys to read
 * @arg prefix_len The length of the prefix
 * @arg cb The callback function to invoke
 * @arg data Opaque handle passed to the callback
 * @return 0 on success, or the return of the callback.
 */
int art_iter_prefix(art_tree *t, const unsigned char *key, int key_len, art_callback cb, void *data)
{
    art_node **child;
    art_node *n = t->root;
    int prefix_len, depth = 0;
    while (n)
    {
        // Might be a leaf
        if (IS_LEAF(n))
        {
            n = (art_node *)LEAF_RAW(n);
            // Check if the expanded path matches
            if (!leaf_prefix_matches((art_leaf *)n, key, key_len))
            {
                art_leaf *l = (art_leaf *)n;
                return cb(data, (const unsigned char *)l->key, l->key_len, l->value);
            }
            return 0;
        }

        // If the depth matches the prefix, we need to handle this node
        if (depth == key_len)
        {
            art_leaf *l = minimum(n);
            if (!leaf_prefix_matches(l, key, key_len))
                return recursive_iter(n, cb, data);
            return 0;
        }

        // Bail if the prefix does not match
        if (n->partial_len)
        {
            prefix_len = prefix_mismatch(n, key, key_len, depth);

            // Guard if the mis-match is longer than the MAX_PREFIX_LEN
            if ((uint32_t)prefix_len > n->partial_len)
            {
                prefix_len = n->partial_len;
            }

            // If there is no match, search is terminated
            if (!prefix_len)
            {
                return 0;

                // If we've matched the prefix, iterate on this node
            }
            else if (depth + prefix_len == key_len)
            {
                return recursive_iter(n, cb, data);
            }

            // if there is a full match, go deeper
            depth = depth + n->partial_len;
        }

        // Recursively search
        child = find_child(n, key[depth]);
        n = (child) ? *child : NULL;
        depth++;
    }
    return 0;
}
void init_region(void **base, size_t size, int use_cxl, struct memkind **kind)
{
    *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(*base != MAP_FAILED);

    unsigned long node_mask = 1UL << (use_cxl ? CXL_MASK : LOCAL_MASK);
    long mbind_ret = mbind(*base, size, MPOL_BIND, &node_mask, sizeof(node_mask) * 8, 0);
    if (mbind_ret != 0)
    {
        perror("mbind");
        abort();
    }
    // printf("Creating fixed kind at %p, size = %zu\n", *base, size);
    // assert(*base != MAP_FAILED);
    // assert(size % sysconf(_SC_PAGESIZE) == 0);
    int err = memkind_create_fixed(*base, size, kind);
    if (err)
    {
        char msg[256];
        memkind_error_message(err, msg, sizeof(msg));
        fprintf(stderr, "memkind_create_fixed failed: %s\n", msg);
        abort();
    }
}

void destroy_region(void *base, size_t size, struct memkind *kind)
{
    int err = memkind_destroy_kind(kind);
    if (err)
    {
        perror("memkind_destroy_kind failed");
        abort();
    }
    munmap(base, size);
}

#if CNT
void node_cnt_stat()
{
    printf("------- Node Count Summry -------\n");
    printf("Node4: %lu\n", node4_cnt);
    printf("Node16: %lu\n", node16_cnt);
    printf("Node48: %lu\n", node48_cnt);
    printf("Node256: %lu\n", node256_cnt);
    printf("Leaf: %lu\n", leaf_cnt);
}
#endif

#if HIT_CNT_TOTAL
void node_hit_cnt_total()
{
    printf("------- Node Hit Count Summry -------\n");
    printf("Node4: %lu\n", node4_hit_cnt);
    printf("Node16: %lu\n", node16_hit_cnt);
    printf("Node48: %lu\n", node48_hit_cnt);
    printf("Node256: %lu\n", node256_hit_cnt);
    printf("Leaf: %lu\n", leaf_hit_cnt);
}
void reset_node_hit_cnt_total()
{
    printf("------- Resetting node hit count total -------\n");
    node4_hit_cnt = node16_hit_cnt = node48_hit_cnt = node256_hit_cnt = leaf_hit_cnt = 0;
}
#endif

#if HIT_DIST
void stream_node_hit_counts_individual(art_node *n, FILE *out)
{
    if (!n)
        return;
    if (IS_LEAF(n))
        return;

    switch (n->type)
    {
    case NODE4:
    {
        fprintf(out, "%lu,4,%d\n", (unsigned long)n, n->hit_cnt);
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            stream_node_hit_counts_individual(node->children[i], out);
        break;
    }
    case NODE16:
    {
        fprintf(out, "%lu,16,%d\n", (unsigned long)n, n->hit_cnt);
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            stream_node_hit_counts_individual(node->children[i], out);
        break;
    }
    case NODE48:
    {
        fprintf(out, "%lu,48,%d\n", (unsigned long)n, n->hit_cnt);
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            uint8_t idx = node->keys[i];
            if (idx)
                stream_node_hit_counts_individual(node->children[idx - 1], out);
        }
        break;
    }
    case NODE256:
    {
        fprintf(out, "%lu,256,%d\n", (unsigned long)n, n->hit_cnt);
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
                stream_node_hit_counts_individual(node->children[i], out);
        }
        break;
    }
    default:
        abort();
    }
}
void cooling_node_hit_cnt_individual(art_node *n, float factor)
{
    if (!n)
        return;
    if (IS_LEAF(n))
        return;

    n->hit_cnt *= factor;
    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            cooling_node_hit_cnt_individual(node->children[i], factor);
        break;
    }
    case NODE16:
    {
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            cooling_node_hit_cnt_individual(node->children[i], factor);
        break;
    }
    case NODE48:
    {
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            uint8_t idx = node->keys[i];
            if (idx)
                cooling_node_hit_cnt_individual(node->children[idx - 1], factor);
        }
        break;
    }
    case NODE256:
    {
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
                cooling_node_hit_cnt_individual(node->children[i], factor);
        }
        break;
    }
    default:
        abort();
    }
}

#endif
#if DEPTH
void collect_node_depths(art_node *n, int depth, node_depth_stats_t *stats, FILE *fd)
{
    if (!n)
        return;
    if (IS_LEAF(n))
        return;

    switch (n->type)
    {
    case NODE4:
    {
        // fprintf(fd, "4: %d\n", n->depth);
        stats->node4_depth_total += depth;
        stats->node4_count++;
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
        {
            collect_node_depths(node->children[i], depth + 1, stats, fd);
        }
        break;
    }

    case NODE16:
    {
        // fprintf(fd, "16: %d\n", n->depth);
        stats->node16_depth_total += depth;
        stats->node16_count++;
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
        {
            collect_node_depths(node->children[i], depth + 1, stats, fd);
        }
        break;
    }

    case NODE48:
    {
        // fprintf(fd, "48: %d\n", n->depth);
        stats->node48_depth_total += depth;
        stats->node48_count++;
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            uint8_t idx = node->keys[i];
            if (idx)
            {
                collect_node_depths(node->children[idx - 1], depth + 1, stats, fd);
            }
        }
        break;
    }

    case NODE256:
    {
        // fprintf(fd, "256: %d\n", n->depth);
        stats->node256_depth_total += depth;
        stats->node256_count++;
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
            {
                collect_node_depths(node->children[i], depth + 1, stats, fd);
            }
        }
        break;
    }

    default:
        abort();
    }
}

void print_avg_node_depths(const node_depth_stats_t *s)
{
    printf("------- Node STAT -------\n");
    if (s->node4_count)
        printf("Node4 total: %zu, avg depth:    %.2f\n", s->node4_count, (double)s->node4_depth_total / s->node4_count);
    if (s->node16_count)
        printf("Node16 total: %zu, avg depth:   %.2f\n", s->node16_count, (double)s->node16_depth_total / s->node16_count);
    if (s->node48_count)
        printf("Node48 total: %zu, avg depth:   %.2f\n", s->node48_count, (double)s->node48_depth_total / s->node48_count);
    if (s->node256_count)
        printf("Node256 total: %zu, avg depth:  %.2f\n", s->node256_count, (double)s->node256_depth_total / s->node256_count);
}
#endif

#if LEVEL_ORDER
void stream_level_distribution(art_node *root, FILE *out)
{
    if (!root)
        return;

    typedef struct node_queue
    {
        art_node *node;
        int depth;
        struct node_queue *next;
    } node_queue;

    node_queue *head = malloc(sizeof(node_queue));
    head->node = root;
    head->depth = 0;
    head->next = NULL;
    node_queue *tail = head;

    int curr_depth = 0;
    int count_4 = 0, count_16 = 0, count_48 = 0, count_256 = 0, count_leaf = 0;
    int hit_cnt_4 = 0, hit_cnt_16 = 0, hit_cnt_48 = 0, hit_cnt_256 = 0, hit_cnt_leaf = 0;

    while (head)
    {
        art_node *n = head->node;
        // n->depth = curr_depth; // cannot do it
        int depth = head->depth;
        if (depth != curr_depth)
        {
            fprintf(out, "LEVEL %d: node4=%d node16=%d node48=%d node256=%d leaf=%d ",
                    curr_depth, count_4, count_16, count_48, count_256, count_leaf);
            fprintf(out, "hit_cnt_4=%d hit_cnt_16=%d hit_cnt_48=%d hit_cnt_256=%d\n",
                    hit_cnt_4, hit_cnt_16, hit_cnt_48, hit_cnt_256);
            curr_depth = depth;
            count_4 = count_16 = count_48 = count_256 = count_leaf = 0;
            hit_cnt_4 = hit_cnt_16 = hit_cnt_48 = hit_cnt_256 = hit_cnt_leaf = 0;
        }

        // Add children of this node to queue
        if (IS_LEAF(n))
        {
            count_leaf++;
        }
        else
        {
            switch (n->type)
            {
            case NODE4:
            {
                hit_cnt_4 += n->hit_cnt;
                count_4++;
                art_node4 *node = (art_node4 *)n;
                for (int i = 0; i < n->num_children; i++)
                {
                    node_queue *entry = malloc(sizeof(node_queue));
                    entry->node = node->children[i];
                    entry->depth = depth + 1;
                    entry->next = NULL;
                    tail->next = entry;
                    tail = entry;
                }
                break;
            }
            case NODE16:
            {
                hit_cnt_16 += n->hit_cnt;
                count_16++;
                art_node16 *node = (art_node16 *)n;
                for (int i = 0; i < n->num_children; i++)
                {
                    node_queue *entry = malloc(sizeof(node_queue));
                    entry->node = node->children[i];
                    entry->depth = depth + 1;
                    entry->next = NULL;
                    tail->next = entry;
                    tail = entry;
                }
                break;
            }
            case NODE48:
            {
                hit_cnt_48 += n->hit_cnt;
                count_48++;
                art_node48 *node = (art_node48 *)n;
                for (int i = 0; i < 256; i++)
                {
                    uint8_t idx = node->keys[i];
                    if (idx)
                    {
                        node_queue *entry = malloc(sizeof(node_queue));
                        entry->node = node->children[idx - 1];
                        entry->depth = depth + 1;
                        entry->next = NULL;
                        tail->next = entry;
                        tail = entry;
                    }
                }
                break;
            }
            case NODE256:
            {
                hit_cnt_256 += n->hit_cnt;
                count_256++;
                art_node256 *node = (art_node256 *)n;
                for (int i = 0; i < 256; i++)
                {
                    if (node->children[i])
                    {
                        node_queue *entry = malloc(sizeof(node_queue));
                        entry->node = node->children[i];
                        entry->depth = depth + 1;
                        entry->next = NULL;
                        tail->next = entry;
                        tail = entry;
                    }
                }
                break;
            }
            default:
                abort();
            }
        }

        node_queue *tmp = head;
        head = head->next;
        free(tmp);
    }

    fprintf(out, "LEVEL %d: node4=%d node16=%d node48=%d node256=%d leaf=%d ",
            curr_depth, count_4, count_16, count_48, count_256, count_leaf);
    fprintf(out, "hit_cnt_4=%d hit_cnt_16=%d hit_cnt_48=%d hit_cnt_256=%d\n",
            hit_cnt_4, hit_cnt_16, hit_cnt_48, hit_cnt_256);
}
#endif

void stream_node_type_addr(art_node *n, FILE *fd)
{
    if (!n)
        return;

    if (IS_LEAF(n))
    {
        // art_leaf *leaf = LEAF_RAW(n);
        // fprintf(fd, "leaf: %p, %ld\n", (void *)leaf, sizeof(*leaf) + leaf->key_len);
        return;
    }
    // fprintf(fd, "4,%lu\n", (unsigned long)n);
    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *node = (art_node4 *)n;
        fprintf(fd, "4,%lu,%ld\n", (unsigned long)n, sizeof(*node));
        for (int i = 0; i < node->n.num_children; i++)
            stream_node_type_addr(node->children[i], fd);
        break;
    }

    case NODE16:
    {
        art_node16 *node = (art_node16 *)n;
        fprintf(fd, "16,%lu,%ld\n", (unsigned long)n, sizeof(*node));
        for (int i = 0; i < node->n.num_children; i++)
            stream_node_type_addr(node->children[i], fd);
        break;
    }

    case NODE48:
    {
        art_node48 *node = (art_node48 *)n;
        fprintf(fd, "48,%lu,%ld\n", (unsigned long)n, sizeof(*node));
        for (int i = 0; i < 256; i++)
        {
            uint8_t idx = node->keys[i];
            if (idx)
                stream_node_type_addr(node->children[idx - 1], fd);
        }
        break;
    }

    case NODE256:
    {
        art_node256 *node = (art_node256 *)n;
        fprintf(fd, "256,%lu,%ld\n", (unsigned long)n, sizeof(*node));
        for (int i = 0; i < 256; i++)
            if (node->children[i])
                stream_node_type_addr(node->children[i], fd);
        break;
    }

    default:
        abort();
    }
}

// static int check_numa_node(void *addr)
// {
//     int status;
//     int ret;

//     // Check the NUMA node of the page that contains addr
//     ret = move_pages(0,       // self process
//                      1,       // one page
//                      &addr,   // address
//                      NULL,    // don't move
//                      &status, // output node status
//                      0);      // flags

//     if (ret == -1)
//     {
//         perror("check addr loc failed");
//         return -1;
//     }

//     if (status >= 0)
//     {
//         // printf("Address %p is on NUMA node %d\n", addr, status);
//         return status;
//     }
//     else
//     {
//         // printf("Address %p is not currently mapped (status = %d)\n", addr, status);
//         return -1;
//     }
// }

#if STATIC_DIST
void distribute_nodes(art_node *n, art_node **ref, int curr_depth)
{
    if (!n)
        return;

    if (IS_LEAF(n))
    {
        art_leaf *old_leaf = LEAF_RAW(n);
        uint32_t key_len = old_leaf->key_len;
        art_leaf *new_leaf = NULL;
        if (curr_depth < DEPTH_THRESH)
        {
            new_leaf = (art_leaf *)memkind_calloc(leaf_local_kind, 1, sizeof(art_leaf) + key_len);
            leaf_moved_local++;
        }
        else
        {
            new_leaf = (art_leaf *)memkind_calloc(leaf_cxl_kind, 1, sizeof(art_leaf) + key_len);
            leaf_moved_cxl++;
        }
        memmove(new_leaf, old_leaf, sizeof(art_leaf) + key_len);
        *ref = (art_node *)SET_LEAF(new_leaf); // TODO, recheck
        memkind_free(leaf_kind, old_leaf);     // free old leaf node
        return;
    }

    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *old_node = (art_node4 *)n;
        art_node4 *new_node = NULL;
        if (curr_depth < DEPTH_THRESH)
        {
            new_node = (art_node4 *)memkind_calloc(node4_local_kind, 1, sizeof(art_node4));
            memmove(new_node, old_node, sizeof(art_node4));
            node4_moved_local++;
        }
        else
        {
            new_node = (art_node4 *)memkind_calloc(node4_cxl_kind, 1, sizeof(art_node4));
            memmove(new_node, old_node, sizeof(art_node4));
            node4_moved_cxl++;
        }
        *ref = (art_node *)new_node;
        for (int i = 0; i < new_node->n.num_children; i++)
        {
            distribute_nodes(new_node->children[i], &new_node->children[i], curr_depth + 1);
        }
        memkind_free(node4_kind, old_node);
        break;
    }
    case NODE16:
    {
        art_node16 *old_node = (art_node16 *)n;
        art_node16 *new_node = NULL;
        if (curr_depth < DEPTH_THRESH)
        {
            new_node = (art_node16 *)memkind_calloc(node16_local_kind, 1, sizeof(art_node16));
            memmove(new_node, old_node, sizeof(art_node16));
            node16_moved_local++;
        }
        else
        {
            new_node = (art_node16 *)memkind_calloc(node16_cxl_kind, 1, sizeof(art_node16));
            memmove(new_node, old_node, sizeof(art_node16));
            node16_moved_cxl++;
        }
        *ref = (art_node *)new_node;
        for (int i = 0; i < new_node->n.num_children; i++)
        {
            distribute_nodes(new_node->children[i], &new_node->children[i], curr_depth + 1);
        }
        memkind_free(node16_kind, old_node);
        break;
    }
    case NODE48:
    {
        art_node48 *old_node = (art_node48 *)n;
        art_node48 *new_node = NULL;
        if (curr_depth < DEPTH_THRESH)
        {
            new_node = (art_node48 *)memkind_calloc(node48_local_kind, 1, sizeof(art_node48));
            memmove(new_node, old_node, sizeof(art_node48));
            node48_moved_local++;
        }
        else
        {
            new_node = (art_node48 *)memkind_calloc(node48_cxl_kind, 1, sizeof(art_node48));
            memmove(new_node, old_node, sizeof(art_node48));
            node48_moved_cxl++;
        }
        *ref = (art_node *)new_node;
        for (int i = 0; i < 256; i++)
        {
            uint8_t idx = new_node->keys[i];
            if (idx)
                distribute_nodes(new_node->children[idx - 1], &new_node->children[idx - 1], curr_depth + 1);
        }
        memkind_free(node48_kind, old_node);
        break;
    }
    case NODE256:
    {
        art_node256 *old_node = (art_node256 *)n;
        art_node256 *new_node = NULL;
        if (1)
        {
            new_node = (art_node256 *)memkind_calloc(node256_local_kind, 1, sizeof(art_node256));
            memmove(new_node, old_node, sizeof(art_node256));
            node256_moved_local++;
        }
        else
        { // do not use it
          // new_node = (art_node256 *)memkind_calloc(node256_cxl_kind, 1, sizeof(art_node256));
          // memmove(new_node, old_node, sizeof(art_node256));
          // node256_moved_cxl++;
        }
        *ref = (art_node *)new_node;
        for (int i = 0; i < 256; i++)
        {
            if (new_node->children[i])
                distribute_nodes(new_node->children[i], &new_node->children[i], curr_depth + 1);
        }
        memkind_free(node256_kind, old_node);
        break;
    }
    default:
        abort();
    }
}

void print_node_move_stat()
{
    printf("leaf_moved: local: %d, cxl: %d\n", leaf_moved_local, leaf_moved_cxl);
    printf("node4_moved: local: %d, cxl: %d\n", node4_moved_local, node4_moved_cxl);
    printf("node16_moved: local: %d, cxl: %d\n", node16_moved_local, node16_moved_cxl);
    printf("node48_moved: local: %d, cxl: %d\n", node48_moved_local, node48_moved_cxl);
    printf("node256_moved: local: %d, cxl: %d\n", node256_moved_local, node256_moved_cxl);
}
#endif

#if ONLINE
art_node *tiered_calloc(bool *local_full,
                        memkind_t local_kind, memkind_t cxl_kind,
                        size_t size,
                        void **node_hot_arr, int *hot_count,
                        void **node_cold_arr, int *cold_count)
{
    art_node *ptr = NULL;

    if (!*local_full)
    {
        void *candidate = memkind_calloc(local_kind, 1, size);
        if (candidate)
        {
            ptr = (art_node *)candidate;
            node_hot_arr[*hot_count] = ptr;
            ptr->idx_in_arr = *hot_count;
            ptr->in_local = true;
            (*hot_count)++;
        }
        else
        {
            *local_full = true; // fallback to CXL
        }
    }

    if (!ptr)
    {
        ptr = (art_node *)memkind_calloc(cxl_kind, 1, size);
        if (!ptr)
        {
            perror("tiered_calloc failed for both local and cxl kinds");
            exit(EXIT_FAILURE);
        }
        node_cold_arr[*cold_count] = ptr;
        ptr->idx_in_arr = *cold_count;
        ptr->in_local = false;
        (*cold_count)++;
    }

    return ptr;
}
static int compare_hit_cnt(const void *a, const void *b)
{
    const art_node *na = *(const art_node **)a;
    const art_node *nb = *(const art_node **)b;
    return nb->hit_cnt - na->hit_cnt; // descending order, for ascending is a - b
}
void sort_hotness(void **alloced_nodes, int alloced_cnt)
{
    printf("sorting total %d\n", alloced_cnt);
    qsort(alloced_nodes, alloced_cnt, sizeof(void *), compare_hit_cnt);
}

static void swap_art_nodes(art_node **n0, art_node **n1)
{
    art_node tmp;
    memcpy(&tmp, *n0, sizeof(art_node4));
    memcpy(*n0, *n1, sizeof(art_node4));
    memcpy(*n1, &tmp, sizeof(art_node4));

    // swap the pointers themselves
    art_node *tmp_ptr = *n0;
    *n0 = *n1;
    *n1 = tmp_ptr;
}
static void update_hot_cold_arr(art_node *n, int *local_alloc_cnt, int *cxl_alloc_cnt, void **hot_arr, void **cold_arr)
{
    int idx = n->idx_in_arr;
    int last = 0;
    n->idx_in_arr = -1;
    if (n->in_local)
    {
        last = *local_alloc_cnt - 1;
        if (idx != last)
        {
            hot_arr[idx] = hot_arr[last];
            ((art_node *)hot_arr[idx])->idx_in_arr = idx;
        }
        (*local_alloc_cnt)--;
    }
    else
    {
        last = *cxl_alloc_cnt - 1;
        if (idx != last)
        {
            cold_arr[idx] = cold_arr[last];
            ((art_node *)cold_arr[idx])->idx_in_arr = idx;
        }
        (*cxl_alloc_cnt)--;
    }
}
#endif
