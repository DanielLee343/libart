#include <stdlib.h>
#include <time.h>
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

#define NUM_COLO_ALL_PAGE 1391624

#define NODE4_SIZE 80.05
#define NODE16_SIZE 192.1
#define NODE48_SIZE 768.65
#define NODE256_SIZE 2561

#define NODE4_SENSI 0.21
#define NODE16_SENSI 0.39
#define NODE48_SENSI 0.96
#define NODE256_SENSI 2.4

// email
// #define NUM_4_PAGE 302600
// #define NUM_16_PAGE 203776
// #define NUM_48_PAGE 112896
// #define NUM_256_PAGE 10
// #define NUM_LEAF_PAGE 772096

// email finer control
// #define NUM_4_PAGE 279912
// #define NUM_16_PAGE 187757
// #define NUM_48_PAGE 105770
// #define NUM_256_PAGE 6
// #define NUM_LEAF_PAGE 720000

// email a_ext_10
// #define NUM_4_PAGE 573440
// #define NUM_16_PAGE 396800
// #define NUM_48_PAGE 353792
// #define NUM_256_PAGE 235100
// #define NUM_LEAF_PAGE 2287400

// #define NUM_4_PAGE_LC 57250
// #define NUM_16_PAGE_LC 212941
// #define NUM_48_PAGE_LC 353792
// #define NUM_256_PAGE_LC 235100

// #define NUM_4_PAGE_CXL 573440
// #define NUM_16_PAGE_CXL 289712
// #define NUM_48_PAGE_CXL 3384
// #define NUM_256_PAGE_CXL 10

// email c_ext_10
#define NUM_4_PAGE 496396
#define NUM_16_PAGE 240034
#define NUM_48_PAGE 247328
#define NUM_256_PAGE 1300000
#define NUM_LEAF_PAGE 22874010

#define NUM_4_PAGE_LC 496396
#define NUM_16_PAGE_LC 240034
#define NUM_48_PAGE_LC 247328
#define NUM_256_PAGE_LC 1300000

#define NUM_4_PAGE_CXL 496396
#define NUM_16_PAGE_CXL 240034
#define NUM_48_PAGE_CXL 247328
#define NUM_256_PAGE_CXL 1300000
// #define NUM_4_PAGE_CXL (NUM_4_PAGE - NUM_4_PAGE_LC)
// #define NUM_16_PAGE_CXL (NUM_16_PAGE - NUM_16_PAGE_LC)
// #define NUM_48_PAGE_CXL (NUM_48_PAGE - NUM_48_PAGE_LC)
// #define NUM_256_PAGE_CXL (NUM_256_PAGE - NUM_256_PAGE_LC)

// randint
// #define NUM_LEAF_PAGE 2930000
// #define NUM_4_PAGE 1664000
// #define NUM_16_PAGE 480000
// #define NUM_48_PAGE 256
// #define NUM_256_PAGE 300000

#define STR(x) #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))

#ifndef NODE4_CUTOFF
#define NODE4_CUTOFF 50
#endif
#ifndef NODE16_CUTOFF
#define NODE16_CUTOFF 50
#endif
#ifndef NODE48_CUTOFF
#define NODE48_CUTOFF 50
#endif
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

#if CLFLUSH4 || CLFLUSH16 || CLFLUSH48 || CLFLUSH256
#define CACHE_LINE_SIZE 64
static inline void clflush(volatile void *p)
{
    __asm__ volatile("clflush (%0)" ::"r"(p));
}

static void clflush_range(void *addr, size_t size)
{
    uintptr_t p = (uintptr_t)addr;
    uintptr_t end = p + size;
    p &= ~(CACHE_LINE_SIZE - 1);

    for (; p < end; p += CACHE_LINE_SIZE)
    {
        clflush((void *)p);
    }
    __asm__ volatile("mfence" ::: "memory");
}
#endif
static const int type_map[] = {0, 4, 16, 48, 256}; // index 0 unused
static double elapsed_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

/**
 * Allocates a node of the given type,
 * initializes to zero and sets the type.
 */
#if STATIC
static art_node *alloc_node(uint8_t type, uint32_t depth)
#else
static art_node *alloc_node(uint8_t type)
#endif
{
    /* type ranging [1,4] but in matrix it's [0,3] so it's type - 1*/
    art_node *n;
    switch (type)
    {
    case NODE4:
// n = ALIGN_UP(node4_ptr, LEAF_ALIGN); // for prev bulk alloced region
// node4_ptr = (void *)((uintptr_t)n + sizeof(art_node4));
// n = (art_node *)calloc(1, sizeof(art_node4)); // vanilla
#if ONLINE
        n = (art_node *)tiered_calloc(&node4_local_full, node4_local_kind, node4_cxl_kind, sizeof(art_node4), node4_hot, &node4_local_alloc_cnt, node4_cold, &node4_cxl_alloc_cnt);
        // n = (art_node *)tiered_calloc(&node4_local_full, node4_local_kind, node4_cxl_kind, sizeof(art_node4), type);
#elif STATIC
        if (matrix[depth][type - 1])
        {
            n = (art_node *)memkind_calloc(node4_local_kind, 1, sizeof(art_node4));
            assert(n);
#if ENABLE_PROFILE
            node4_local_cnt++;
#endif
        }
        else
        {
            n = (art_node *)memkind_calloc(node4_cxl_kind, 1, sizeof(art_node4));
            assert(n);
#if ENABLE_PROFILE
            node4_cxl_cnt++;
#endif
        }
#else
        n = (art_node *)memkind_calloc(node4_kind, 1, sizeof(art_node4));
        assert(n);
#endif
#if CNT
        node4_cnt++;
#endif
        break;
    case NODE16:
#if ONLINE
        n = (art_node *)tiered_calloc(&node16_local_full, node16_local_kind, node16_cxl_kind, sizeof(art_node16), node16_hot, &node16_local_alloc_cnt, node16_cold, &node16_cxl_alloc_cnt);
        // n = (art_node *)tiered_calloc(&node16_local_full, node16_local_kind, node16_cxl_kind, sizeof(art_node16), type);
#elif STATIC
        if (matrix[depth][type - 1])
        {
            n = (art_node *)memkind_calloc(node16_local_kind, 1, sizeof(art_node16));
            assert(n);
#if ENABLE_PROFILE
            node16_local_cnt++;
#endif
        }
        else
        {
            n = (art_node *)memkind_calloc(node16_cxl_kind, 1, sizeof(art_node16));
            assert(n);
#if ENABLE_PROFILE
            node16_cxl_cnt++;
#endif
        }
#else
        n = (art_node *)memkind_calloc(node16_kind, 1, sizeof(art_node16));
        assert(n);
#endif
#if CNT
        node16_cnt++;
#endif
        break;
    case NODE48:
#if ONLINE
        n = (art_node *)tiered_calloc(&node48_local_full, node48_local_kind, node48_cxl_kind, sizeof(art_node48), node48_hot, &node48_local_alloc_cnt, node48_cold, &node48_cxl_alloc_cnt);
        // n = (art_node *)tiered_calloc(&node48_local_full, node48_local_kind, node48_cxl_kind, sizeof(art_node48), type);
#elif STATIC
        if (matrix[depth][type - 1])
        {
            n = (art_node *)memkind_calloc(node48_local_kind, 1, sizeof(art_node48));
            assert(n);
#if ENABLE_PROFILE
            node48_local_cnt++;
#endif
        }
        else
        {
            n = (art_node *)memkind_calloc(node48_cxl_kind, 1, sizeof(art_node48));
            assert(n);
#if ENABLE_PROFILE
            node48_cxl_cnt++;
#endif
        }
#else
        n = (art_node *)memkind_calloc(node48_kind, 1, sizeof(art_node48));
        assert(n);
#endif
#if CNT
        node48_cnt++;
#endif
        break;
    case NODE256:
#if ONLINE
        n = (art_node *)tiered_calloc(&node256_local_full, node256_local_kind, node256_cxl_kind, sizeof(art_node256), node256_hot, &node256_local_alloc_cnt, node256_cold, &node256_cxl_alloc_cnt);
        // n = (art_node *)tiered_calloc(&node256_local_full, node256_local_kind, node256_cxl_kind, sizeof(art_node256), type);
#elif STATIC
        if (matrix[depth][type - 1])
        {
            n = (art_node *)memkind_calloc(node256_local_kind, 1, sizeof(art_node256));
            assert(n);
#if ENABLE_PROFILE
            node256_local_cnt++;
#endif
        }
        else
        {
            n = (art_node *)memkind_calloc(node256_cxl_kind, 1, sizeof(art_node256));
            assert(n);
#if ENABLE_PROFILE
            node256_cxl_cnt++;
#endif
        }
#else
        n = (art_node *)memkind_calloc(node256_kind, 1, sizeof(art_node256));
        assert(n);
#endif
#if CNT
        node256_cnt++;
#endif
        break;
    default:
        abort();
    }
    n->type = type;
#if STREAM_ACC_ADDR
    fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
#endif
    return n;
}
#if STATIC
static art_node *deep_copy_and_replace(art_node *n, struct memkind *target_kind)
{
    assert(n != NULL);
    art_node **ref = n->self_ref;
    assert(ref != NULL);

    art_node *new_n = NULL;
    struct memkind *old_kind = memkind_detect_kind((void *)n);

    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *old_node = (art_node4 *)n;
        art_node4 *new_node = (art_node4 *)memkind_calloc(target_kind, 1, sizeof(art_node4));
        memcpy(new_node, old_node, sizeof(art_node4));

        new_node->n.self_ref = ref;
        for (int i = 0; i < new_node->n.num_children; ++i)
        {
            art_node *child = new_node->children[i];
            if (!IS_LEAF(child))
                child->self_ref = &new_node->children[i];
        }

        new_n = (art_node *)new_node;
        break;
    }

    case NODE16:
    {
        art_node16 *old_node = (art_node16 *)n;
        art_node16 *new_node = (art_node16 *)memkind_calloc(target_kind, 1, sizeof(art_node16));
        memcpy(new_node, old_node, sizeof(art_node16));

        new_node->n.self_ref = ref;
        for (int i = 0; i < new_node->n.num_children; ++i)
        {
            art_node *child = new_node->children[i];
            if (!IS_LEAF(child))
                child->self_ref = &new_node->children[i];
        }

        new_n = (art_node *)new_node;
        break;
    }

    case NODE48:
    {
        art_node48 *old_node = (art_node48 *)n;
        art_node48 *new_node = (art_node48 *)memkind_calloc(target_kind, 1, sizeof(art_node48));
        memcpy(new_node, old_node, sizeof(art_node48));

        new_node->n.self_ref = ref;
        for (int i = 0; i < 256; ++i)
        {
            if (new_node->keys[i])
            {
                int pos = new_node->keys[i] - 1;
                art_node *child = new_node->children[pos];
                if (!IS_LEAF(child))
                    child->self_ref = &new_node->children[pos];
            }
        }

        new_n = (art_node *)new_node;
        break;
    }

    case NODE256:
    {
        art_node256 *old_node = (art_node256 *)n;
        art_node256 *new_node = (art_node256 *)memkind_calloc(target_kind, 1, sizeof(art_node256));
        memcpy(new_node, old_node, sizeof(art_node256));

        new_node->n.self_ref = ref;
        for (int i = 0; i < 256; ++i)
        {
            art_node *child = new_node->children[i];
            if (child && !IS_LEAF(child))
                child->self_ref = &new_node->children[i];
        }

        new_n = (art_node *)new_node;
        break;
    }

    default:
        abort();
    }

    *ref = new_n;
    memkind_free(old_kind, n);
    return new_n;
}

static art_node *try_copy_node(art_node *n)
{
    int type_index = n->type - 1;
    int should_be_local = matrix[n->depth][type_index]; // bug
    struct memkind *actual_kind = memkind_detect_kind((void *)n);
    art_node *new_node;
    if (should_be_local && actual_kind == cxl_kinds[type_index])
    {
        new_node = deep_copy_and_replace(n, local_kinds[type_index]);
#if ENABLE_PROFILE
        (*misplaced_cnt[type_index])++;
        (*local_cnt[type_index])++;
        (*cxl_cnt[type_index])--;
#endif
    }
    else if (!should_be_local && actual_kind == local_kinds[type_index])
    {
        new_node = deep_copy_and_replace(n, cxl_kinds[type_index]);
#if ENABLE_PROFILE
        (*misplaced_cnt[type_index])++;
        (*local_cnt[type_index])--;
        (*cxl_cnt[type_index])++;
#endif
    }
    else
    { // nodes correctly placed, just return current node
        return n;
    }
    return new_node;
}
#endif
#if DEPTH_INDI
static void increment_subtree_depth(art_node *n)
{
    if (!n)
        return;

    if (IS_LEAF(n))
    {
        return;
    }
    total_subtree_incremented_nodes++;
    n->depth += 1;

    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            increment_subtree_depth(node->children[i]);
#if STATIC
        try_copy_node(n);
#endif
        break;
    }
    case NODE16:
    {
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            increment_subtree_depth(node->children[i]);
#if STATIC
        try_copy_node(n);
#endif
        break;
    }
    case NODE48:
    {
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->keys[i])
            {
                int idx = node->keys[i] - 1;
                increment_subtree_depth(node->children[idx]);
            }
        }
#if STATIC
        try_copy_node(n);
#endif
        break;
    }
    case NODE256:
    {
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
                increment_subtree_depth(node->children[i]);
        }
#if STATIC
        try_copy_node(n);
#endif
        break;
    }
    }
}
#endif

/**
 * Initializes an ART tree
 * @return 0 on success.
 */
int art_tree_init(art_tree *t, char *wl)
{
    SHOW_DEFINE(NODE4_CXL);
    SHOW_DEFINE(NODE16_CXL);
    SHOW_DEFINE(NODE48_CXL);
    SHOW_DEFINE(NODE256_CXL);
    if (CLFLUSH4 || CLFLUSH16 || CLFLUSH48 || CLFLUSH256)
    {
        SHOW_DEFINE(CLFLUSH4);
        SHOW_DEFINE(CLFLUSH16);
        SHOW_DEFINE(CLFLUSH48);
        SHOW_DEFINE(CLFLUSH256);
    }
    t->root = NULL;
    t->size = 0;
    log_fd = fopen("/home/lyuze/workspace/libart/tests/out.log", "w");
#if ENABLE_PROFILE
#endif
#if STATIC || ONLINE
    init_region(&node4_local, (size_t)PAGE_SIZE * NUM_4_PAGE_LC, 0, &node4_local_kind);
    init_region(&node4_cxl, (size_t)PAGE_SIZE * NUM_4_PAGE_CXL, 1, &node4_cxl_kind);
    init_region(&node16_local, (size_t)PAGE_SIZE * NUM_16_PAGE_LC, 0, &node16_local_kind);
    init_region(&node16_cxl, (size_t)PAGE_SIZE * NUM_16_PAGE_CXL, 1, &node16_cxl_kind);
    init_region(&node48_local, (size_t)PAGE_SIZE * NUM_48_PAGE_LC, 0, &node48_local_kind);
    init_region(&node48_cxl, (size_t)PAGE_SIZE * NUM_48_PAGE_CXL, 1, &node48_cxl_kind);
    init_region(&node256_local, (size_t)PAGE_SIZE * NUM_256_PAGE_LC, 0, &node256_local_kind);
    init_region(&node256_cxl, (size_t)PAGE_SIZE * NUM_256_PAGE_CXL, 1, &node256_cxl_kind);
    local_kinds[0] = node4_local_kind;
    local_kinds[1] = node16_local_kind;
    local_kinds[2] = node48_local_kind;
    local_kinds[3] = node256_local_kind;
    cxl_kinds[0] = node4_cxl_kind;
    cxl_kinds[1] = node16_cxl_kind;
    cxl_kinds[2] = node48_cxl_kind;
    cxl_kinds[3] = node256_cxl_kind;
#if ENABLE_PROFILE
    local_cnt[0] = &node4_local_cnt;
    local_cnt[1] = &node16_local_cnt;
    local_cnt[2] = &node48_local_cnt;
    local_cnt[3] = &node256_local_cnt;
    cxl_cnt[0] = &node4_cxl_cnt;
    cxl_cnt[1] = &node16_cxl_cnt;
    cxl_cnt[2] = &node48_cxl_cnt;
    cxl_cnt[3] = &node256_cxl_cnt;
    misplaced_cnt[0] = &node4_mis_placed;
    misplaced_cnt[1] = &node16_mis_placed;
    misplaced_cnt[2] = &node48_mis_placed;
    misplaced_cnt[3] = &node256_mis_placed;
    for (int i = 0; i < 4; i++)
    {
        assert(local_kinds[i] != NULL);
        assert(cxl_kinds[i] != NULL);
        assert(local_cnt[i] != NULL);
        assert(cxl_cnt[i] != NULL);
        assert(misplaced_cnt[i] != NULL);
    }
#endif
#else
    init_region(&node4_base, (size_t)PAGE_SIZE * NUM_4_PAGE, NODE4_CXL, &node4_kind);
    init_region(&node16_base, (size_t)PAGE_SIZE * NUM_16_PAGE, NODE16_CXL, &node16_kind);
    init_region(&node48_base, (size_t)PAGE_SIZE * NUM_48_PAGE, NODE48_CXL, &node48_kind);
    init_region(&node256_base, (size_t)PAGE_SIZE * NUM_256_PAGE, NODE256_CXL, &node256_kind);
#endif
    init_region(&leaf_base, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, LEAF_CXL, &leaf_kind); // leaf not consider tiering
#if ONLINE
    // metadata for online swapping
    node4_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_4_PAGE) / NODE4_SIZE) * sizeof(void *));
    node4_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_4_PAGE) / NODE4_SIZE) * sizeof(void *));
    node16_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_16_PAGE) / NODE16_SIZE) * sizeof(void *));
    node16_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_16_PAGE) / NODE16_SIZE) * sizeof(void *));
    node48_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_48_PAGE) / NODE48_SIZE) * sizeof(void *));
    node48_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_48_PAGE) / NODE48_SIZE) * sizeof(void *));
    node256_hot = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_256_PAGE) / NODE256_SIZE) * sizeof(void *));
    node256_cold = (void *)calloc(1, (((size_t)PAGE_SIZE * NUM_256_PAGE) / NODE256_SIZE) * sizeof(void *));
    // hot_cache_reset();
    reset_ring_buffer();
#endif
#if STATIC
    load_static_metrics(wl);
#endif
    printf("art_node size: %zu\n", sizeof(art_node));
    printf("node4 size: %zu\n", sizeof(art_node4));
    printf("node16 size: %zu\n", sizeof(art_node16));
    printf("node48 size: %zu\n", sizeof(art_node48));
    printf("node256 size: %zu\n", sizeof(art_node256));
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
#if ENABLE_PROFILE
        if (kind == node4_local_kind)
        {
            node4_local_cnt--;
        }
        else
        {
            node4_cxl_cnt--;
        }
#endif
#if CNT
        node4_cnt--;
#endif
#if ONLINE
        // del_node_from_arr(n, kind);
        del_node_from_arr(n, &node4_local_alloc_cnt, &node4_cxl_alloc_cnt, node4_hot, node4_cold);
#endif
        break;

    case NODE16:
        p.p2 = (art_node16 *)n;
        for (i = 0; i < n->num_children; i++)
        {
            destroy_node(p.p2->children[i]);
        }
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node16_local_kind)
        {
            node16_local_cnt--;
        }
        else
        {
            node16_cxl_cnt--;
        }
#endif
#if CNT
        node16_cnt--;
#endif
#if ONLINE
        // del_node_from_arr(n, kind);
        del_node_from_arr(n, &node16_local_alloc_cnt, &node16_cxl_alloc_cnt, node16_hot, node16_cold);
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
#if ENABLE_PROFILE
        if (kind == node48_local_kind)
        {
            node48_local_cnt--;
        }
        else
        {
            node48_cxl_cnt--;
        }
#endif
#if CNT
        node48_cnt--;
#endif
#if ONLINE
        // del_node_from_arr(n, kind);
        del_node_from_arr(n, &node48_local_alloc_cnt, &node48_cxl_alloc_cnt, node48_hot, node48_cold);
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
#if ENABLE_PROFILE
        if (kind == node256_local_kind)
        {
            node256_local_cnt--;
        }
        else
        {
            node256_cxl_cnt--;
        }
#endif
#if CNT
        node256_cnt--;
#endif
#if ONLINE
        // del_node_from_arr(n, kind);
        del_node_from_arr(n, &node256_local_alloc_cnt, &node256_cxl_alloc_cnt, node256_hot, node256_cold);
#endif
        break;

    default:
        abort();
    }
    // free(n);
}
int show_stat()
{
#if ENABLE_PROFILE
    printf("subtree_inc_func_called: %zu, total_subtree_incremented_nodes: %zu\n", subtree_inc_func_called, total_subtree_incremented_nodes);
    printf("place\tmis\n");
    printf("node4\t%d\n", node4_mis_placed);
    printf("node16\t%d\n", node16_mis_placed);
    printf("node48\t%d\n", node48_mis_placed);
    printf("node256\t%d\n", node256_mis_placed);
    float node4_mb_local = node4_local_cnt * NODE4_SIZE / (1024 * 1024);
    float node4_mb_cxl = node4_cxl_cnt * NODE4_SIZE / (1024 * 1024);
    float node16_mb_local = node16_local_cnt * NODE16_SIZE / (1024 * 1024);
    float node16_mb_cxl = node16_cxl_cnt * NODE16_SIZE / (1024 * 1024);
    float node48_mb_local = node48_local_cnt * NODE48_SIZE / (1024 * 1024);
    float node48_mb_cxl = node48_cxl_cnt * NODE48_SIZE / (1024 * 1024);
    float node256_mb_local = node256_local_cnt * NODE256_SIZE / (1024 * 1024);
    float node256_mb_cxl = node256_cxl_cnt * NODE256_SIZE / (1024 * 1024);

    printf("4 local: %d, %.2f MB, %.2f pages\n", node4_local_cnt, node4_mb_local, ceil(node4_mb_local * 256));
    printf("4 cxl: %d, %.2f MB, %.2f pages\n", node4_cxl_cnt, node4_mb_cxl, ceil(node4_mb_cxl * 256));
    printf("16 local: %d, %.2f MB, %.2f pages\n", node16_local_cnt, node16_mb_local, ceil(node16_mb_local * 256));
    printf("16 cxl: %d, %.2f MB, %.2f pages\n", node16_cxl_cnt, node16_mb_cxl, ceil(node16_mb_cxl * 256));
    printf("48 local: %d, %.2f MB, %.2f pages\n", node48_local_cnt, node48_mb_local, ceil(node48_mb_local * 256));
    printf("48 cxl: %d, %.2f MB, %.2f pages\n", node48_cxl_cnt, node48_mb_cxl, ceil(node48_mb_cxl * 256));
    printf("256 local: %d, %.2f MB, %.2f pages\n", node256_local_cnt, node256_mb_local, ceil(node256_mb_local * 256));
    printf("256 cxl: %d, %.2f MB, %.2f pages\n", node256_cxl_cnt, node256_mb_cxl, ceil(node256_mb_cxl * 256));
#endif
}

/**
 * Destroys an ART tree
 * @return 0 on success.
 */
int art_tree_destroy(art_tree *t)
{
    destroy_node(t->root);
#if STATIC || ONLINE
    destroy_region(node4_local, (size_t)PAGE_SIZE * NUM_4_PAGE_LC, node4_local_kind);
    destroy_region(node4_cxl, (size_t)PAGE_SIZE * NUM_4_PAGE_CXL, node4_cxl_kind);
    destroy_region(node16_local, (size_t)PAGE_SIZE * NUM_16_PAGE_LC, node16_local_kind);
    destroy_region(node16_cxl, (size_t)PAGE_SIZE * NUM_16_PAGE_CXL, node16_cxl_kind);
    destroy_region(node48_local, (size_t)PAGE_SIZE * NUM_48_PAGE_LC, node48_local_kind);
    destroy_region(node48_cxl, (size_t)PAGE_SIZE * NUM_48_PAGE_CXL, node48_cxl_kind);
    destroy_region(node256_local, (size_t)PAGE_SIZE * NUM_256_PAGE_LC, node256_local_kind);
    destroy_region(node256_cxl, (size_t)PAGE_SIZE * NUM_256_PAGE_CXL, node256_cxl_kind);
#else
    destroy_region(node4_base, (size_t)PAGE_SIZE * NUM_4_PAGE, node4_kind);
    destroy_region(node16_base, (size_t)PAGE_SIZE * NUM_16_PAGE, node16_kind);
    destroy_region(node48_base, (size_t)PAGE_SIZE * NUM_48_PAGE, node48_kind);
    destroy_region(node256_base, (size_t)PAGE_SIZE * NUM_256_PAGE, node256_kind);
#endif
    destroy_region(leaf_base, (size_t)PAGE_SIZE * NUM_LEAF_PAGE, leaf_kind); // leaf not consider tiering
    fclose(log_fd);

#if ONLINE
    free(node4_hot);
    free(node4_cold);
    free(node16_hot);
    free(node16_cold);
    free(node48_hot);
    free(node48_cold);
    free(node256_hot);
    free(node256_cold);
#endif
#if STATIC
    free_static_metrics();
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
//  */
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
            // #if STREAM_ACC_ADDR
            //             if (start_acc_streaming)
            //                 fprintf(acc_fd, "%lu,0\n", (unsigned long)n);
            // #endif
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
#if CLFLUSH4 || CLFLUSH16 || CLFLUSH48 || CLFLUSH256
        switch (n->type)
        {
#if CLFLUSH4
        case NODE4:
            clflush_range((void *)n, sizeof(art_node4));
            break;
#endif
#if CLFLUSH16
        case NODE16:
            clflush_range((void *)n, sizeof(art_node16));
            break;
#endif
#if CLFLUSH48
        case NODE48:
            clflush_range((void *)n, sizeof(art_node48));
            break;
#endif
#if CLFLUSH256
        case NODE256:
            clflush_range((void *)n, sizeof(art_node256));
            break;
#endif
        }
#endif
#if HIT_DIST
        n->hit_cnt++;
#endif
#if ONLINE
        // hot_cache_record_access(n); // too much overhead
        // sample_to_ring_buffer(n);
#endif
        // #if STREAM_ACC_ADDR
        //         if (start_acc_streaming)
        //             fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
        // #endif

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
// void *art_search(const art_tree *t, const unsigned char *key, int key_len)
// {
//     art_node **child;
//     art_node *n = t->root;
//     int prefix_len, depth = 0;
//     while (n)
//     {
//         // Might be a leaf
//         if (IS_LEAF(n))
//         {
//             n = (art_node *)LEAF_RAW(n);
//             // Check if the expanded path matches
//             if (!leaf_matches((art_leaf *)n, key, key_len, depth))
//             {
//                 return ((art_leaf *)n)->value;
//             }
//             return NULL;
//         }

//         // Bail if the prefix does not match
//         if (n->partial_len)
//         {
//             prefix_len = check_prefix(n, key, key_len, depth);
//             if (prefix_len != min(MAX_PREFIX_LEN, n->partial_len))
//                 return NULL;
//             depth = depth + n->partial_len;
//         }

//         // Recursively search
//         child = find_child(n, key[depth]);
//         n = (child) ? *child : NULL;
//         depth++;
//     }
//     return NULL;
// }

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

static art_leaf *make_leaf(const unsigned char *key, int key_len, void *value, int depth)
{
    art_leaf *l = NULL;
    // l = ALIGN_UP(bump_ptr, LEAF_ALIGN); // bulk alloc version
    // bump_ptr = (void *)((uintptr_t)l + sizeof(art_leaf) + key_len);
    // l = (art_leaf *)calloc(1, sizeof(art_leaf) + key_len); // vanilla
    l = (art_leaf *)memkind_calloc(leaf_kind, 1, sizeof(art_leaf) + key_len);
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

#if SELF_REF
static inline void refresh_self_refs(art_node *n, int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        switch (n->type)
        {
        case NODE4:
        {
            art_node4 *n4 = (art_node4 *)n;
            art_node *child = n4->children[i];
            if (!IS_LEAF(child))
                child->self_ref = &n4->children[i];
            break;
        }
        case NODE16:
        {
            art_node16 *n16 = (art_node16 *)n;
            art_node *child = n16->children[i];
            if (!IS_LEAF(child))
                child->self_ref = &n16->children[i];
            break;
        }
        default:
            abort();
        }
    }
}
static void fix_children_self_ref(void **children, int count)
{
#if SELF_REF
    for (int i = 0; i < count; ++i)
    {
        if (!IS_LEAF(children[i]))
        {
            ((art_node *)children[i])->self_ref = (art_node **)&children[i];
        }
    }
#endif
}
#endif
static void copy_header(art_node *dest, art_node *src, art_node **ref)
{
    dest->num_children = src->num_children;
    dest->partial_len = src->partial_len;
#if DEPTH_INDI
    dest->depth = src->depth;
#endif
#if SELF_REF
    dest->self_ref = ref; // should never be leaf
#endif
    memcpy(dest->partial, src->partial, min(MAX_PREFIX_LEN, src->partial_len));
}

static void add_child256(art_node256 *n, art_node **ref, unsigned char c, void *child)
{
    (void)ref;
    n->n.num_children++;
    n->children[c] = (art_node *)child;
#if SELF_REF
    ((art_node *)child)->self_ref = &n->children[c]; // could never be a leaf
#endif
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
#if SELF_REF
        if (!IS_LEAF(child))
        {
            ((art_node *)child)->self_ref = &n->children[pos];
        }
        else
        {
#if LEAF_REF
            LEAF_RAW(child)->self_ref = &n->children[pos];
#endif
        }
#endif
        n->n.num_children++;
    }
    else
    {
#if STATIC
        art_node256 *new_node = (art_node256 *)alloc_node(NODE256, ((art_node *)n)->depth);
#else
        art_node256 *new_node = (art_node256 *)alloc_node(NODE256);
#endif
        for (int i = 0; i < 256; i++)
        {
            if (n->keys[i])
            {
                art_node *existing_child = n->children[n->keys[i] - 1];
                new_node->children[i] = existing_child;
#if SELF_REF
                if (!IS_LEAF(existing_child))
                    existing_child->self_ref = &new_node->children[i];
                else
                {
#if LEAF_REF
                    LEAF_RAW(existing_child)->self_ref = &new_node->children[i];
#endif
                }
#endif
            }
        }
        copy_header((art_node *)new_node, (art_node *)n, ref);
        *ref = (art_node *)new_node;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node48_local_kind)
        {
            node48_local_cnt--;
        }
        else
        {
            node48_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node48_local_alloc_cnt, &node48_cxl_alloc_cnt, node48_hot, node48_cold);
#endif
#if CNT
        node48_cnt--;
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
#if SELF_REF
            refresh_self_refs((art_node *)n, idx + 1, n->n.num_children + 1);
#endif
        }
        else
        {
            idx = n->n.num_children;
        }

        n->keys[idx] = c;
        n->children[idx] = child_node;
#if SELF_REF
        if (!IS_LEAF(child))
        {
            ((art_node *)child)->self_ref = &n->children[idx];
        }
        else
        {
#if LEAF_REF
            LEAF_RAW(child)->self_ref = &n->children[idx];
#endif
        }
#endif
        n->n.num_children++;
    }
    else
    {
#if STATIC
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48, ((art_node *)n)->depth);
#else
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48);
#endif

        // Copy existing children
        memcpy(new_node->children, n->children,
               sizeof(void *) * n->n.num_children);
        for (int i = 0; i < n->n.num_children; i++)
        {
            // unsigned char k = n->keys[i];
            // new_node->keys[k] = i + 1;
            new_node->keys[n->keys[i]] = i + 1;
            // new_node->children[i] = n->children[i];
#if SELF_REF
            if (!IS_LEAF(new_node->children[i]))
                new_node->children[i]->self_ref = &new_node->children[i];
            else
            {
#if LEAF_REF
                LEAF_RAW(new_node->children[i])->self_ref = &new_node->children[i];
#endif
            }
#endif
        }
        copy_header((art_node *)new_node, (art_node *)n, ref);
        *ref = (art_node *)new_node;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node16_local_kind)
        {
            node16_local_cnt--;
        }
        else
        {
            node16_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node16_local_alloc_cnt, &node16_cxl_alloc_cnt, node16_hot, node16_cold);
#endif

#if CNT
        node16_cnt--;
#endif
        add_child48(new_node, ref, c, child);
    }
}
static void add_child4(art_node4 *n, art_node **ref, unsigned char c, void *child)
{
    if (n->n.num_children < 4)
    {
        // printf("child: %p\n", child);
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
#if SELF_REF
        refresh_self_refs((art_node *)n, idx + 1, n->n.num_children + 1);
#endif

        // Insert element
        n->keys[idx] = c;
        n->children[idx] = (art_node *)child;
        n->n.num_children++;
#if SELF_REF
        if (!IS_LEAF(child))
        {
            ((art_node *)child)->self_ref = &n->children[idx];
        }
        else
        {
#if LEAF_REF
            LEAF_RAW(child)->self_ref = &n->children[idx];
#endif
        }
#endif
        // printf("4: parent: %p, child: %p\n", ((art_node *)child)->self_ref, child);
    }
    else
    {
#if STATIC
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16, ((art_node *)n)->depth);
#else
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16);
#endif

        // memcpy(new_node->children, n->children,
        //        sizeof(void *) * n->n.num_children);
        // memcpy(new_node->keys, n->keys,
        //        sizeof(unsigned char) * n->n.num_children);
        for (int i = 0; i < n->n.num_children; i++)
        {
            new_node->keys[i] = n->keys[i];
            new_node->children[i] = n->children[i];
#if SELF_REF
            if (!IS_LEAF(new_node->children[i]))
            {
                ((art_node *)new_node->children[i])->self_ref = &new_node->children[i];
            }
#endif
        }
        // fix_children_self_ref(new_node->children, new_node->n.num_children);
        new_node->n.num_children = n->n.num_children;

        copy_header((art_node *)new_node, (art_node *)n, ref);
        *ref = (art_node *)new_node;
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node4_local_kind)
        {
            node4_local_cnt--;
        }
        else
        {
            node4_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node4_local_alloc_cnt, &node4_cxl_alloc_cnt, node4_hot, node4_cold);
#endif
#if CNT
        node4_cnt--;
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
        add_child16((art_node16 *)n, ref, c, child);
        // printf("%d\n", n->depth);
        return;
    case NODE48:
        add_child48((art_node48 *)n, ref, c, child);
        // printf("%p %d\n", n, n->depth);
        return;
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

static void *recursive_insert(art_node *n, art_node **ref, const unsigned char *key, int key_len, void *value, int depth, int *old, int replace, int logical_depth)
{
    // If we are at a NULL node, inject a leaf
    if (!n)
    {
        art_leaf *l = make_leaf(key, key_len, value, logical_depth); // not affecting
        *ref = (art_node *)SET_LEAF(l);
#if LEAF_REF
        (l)->self_ref = ref;
#endif
        return NULL;
    }

    // If we are at a leaf, we need to replace it with a node
    if (IS_LEAF(n))
    {
#if HIT_CNT_TOTAL
        leaf_hit_cnt++;
#endif
        // #if STREAM_ACC_ADDR
        //         if (start_acc_streaming)
        //             fprintf(acc_fd, "%lu,0\n", (unsigned long)n);
        // #endif
        art_leaf *l = LEAF_RAW(n);
        if (!leaf_matches(l, key, key_len, depth))
        { // exact same match, update value
            *old = 1;
            void *old_val = l->value;
            if (replace)
                l->value = value;
            return old_val;
        }

/* Split the leaf into a new NODE4                            */
#if STATIC
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4, logical_depth);
#else
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
#endif
#if DEPTH_INDI
        new_node->n.depth = logical_depth;
        // printf("aaaaa %s: %d\n", key, logical_depth + 1);
        // l->depth = logical_depth + 1;
#endif
        /* new leaf */
        art_leaf *l2 = make_leaf(key, key_len, value, logical_depth + 1);

        /* longest common prefix of the two leaves */
        int lcp = longest_common_prefix(l, l2, depth);
        new_node->n.partial_len = lcp;
        memcpy(new_node->n.partial,
               key + depth,
               min(MAX_PREFIX_LEN, lcp));

        *ref = (art_node *)new_node;
#if SELF_REF
        new_node->n.self_ref = ref;
#endif
        // printf("l addr = %p ref = %p key = %s\n", l, ref, l->key);
        // printf("l2 addr = %p ref = %p key = %s\n", l2, ref, l2->key);
        add_child4(new_node, ref, l->key[depth + lcp], SET_LEAF(l));
        add_child4(new_node, ref, l2->key[depth + lcp], SET_LEAF(l2));
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
#if CLFLUSH4 || CLFLUSH16 || CLFLUSH48 || CLFLUSH256
    switch (n->type)
    {
#if CLFLUSH4
    case NODE4:
        clflush_range((void *)n, sizeof(art_node4));
        break;
#endif
#if CLFLUSH16
    case NODE16:
        clflush_range((void *)n, sizeof(art_node16));
        break;
#endif
#if CLFLUSH48
    case NODE48:
        clflush_range((void *)n, sizeof(art_node48));
        break;
#endif
#if CLFLUSH256
    case NODE256:
        clflush_range((void *)n, sizeof(art_node256));
        break;
#endif
    }
#endif
#if HIT_DIST
    n->hit_cnt++;
#endif
    // #if STREAM_ACC_ADDR
    //     if (start_acc_streaming)
    //         fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
    // #endif
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
#if STATIC
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4, logical_depth);
#else
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
#endif
#if DEPTH_INDI
        new_node->n.depth = logical_depth;
        n->depth = logical_depth + 1;
#if STATIC
        n = try_copy_node(n);
#endif
#endif
        new_node->n.partial_len = prefix_diff;
        memcpy(new_node->n.partial, n->partial, min(MAX_PREFIX_LEN, prefix_diff));
        *ref = (art_node *)new_node;
#if SELF_REF
        new_node->n.self_ref = ref;
        // printf("22222 %p\n", new_node->n.self_ref);
#endif

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
        // printf("bbbbb %s: %d\n", key, logical_depth + 1);
        // Insert the new leaf
        art_leaf *l = make_leaf(key, key_len, value, logical_depth + 1);
        add_child4(new_node, ref, key[depth + prefix_diff], SET_LEAF(l)); // here new_node (node4) will be parent of n (eg., node48)
#if DEPTH_INDI
        // if (n->type == NODE48 || n->type == NODE16 || n->type == NODE4)
        {
            // printf("new_node: %p, n: %p\n", new_node, n);
            // do recursive bump update
            n->depth--; // since we're incrementing inside
            subtree_inc_func_called++;
            increment_subtree_depth(n);
        }
#endif
        return NULL;
    }

RECURSE_SEARCH:;
    art_node **child = find_child(n, key[depth]);
    if (child)
    {
        // printf("ccccc %s: %d\n", key, logical_depth + 1);
        return recursive_insert(*child, child, key, key_len, value, depth + 1, old, replace, logical_depth + 1);
    }
    // No child, node goes within us
    // printf("ddddd %s: %d\n", key, logical_depth);
    art_leaf *l = make_leaf(key, key_len, value, logical_depth);
    // printf("%p\n", LEAF_RAW(l));
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
    void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0, &old_val, 1, 0);
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
    void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0, &old_val, 0, 0);
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
#if STATIC
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48, ((art_node *)n)->depth);
#else
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48);
#endif
        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n, ref);
        int pos = 0;
        for (int i = 0; i < 256; i++)
        {
            if (n->children[i])
            {
                art_node *child = n->children[i];
                new_node->children[pos] = child;
                new_node->keys[i] = pos + 1;
#if SELF_REF
                if (!IS_LEAF(child))
                    child->self_ref = &new_node->children[pos];
                else
#if LEAF_REF
                    LEAF_RAW(child)->self_ref = &new_node->children[pos];
#endif
#endif
                pos++;
            }
        }
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node256_local_kind)
        {
            node256_local_cnt--;
        }
        else
        {
            node256_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node256_local_alloc_cnt, &node256_cxl_alloc_cnt, node256_hot, node256_cold);
#endif
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
#if STATIC
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16, ((art_node *)n)->depth);
#else
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16);
#endif
        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n, ref);
        int child = 0;
        for (int i = 0; i < 256; i++)
        {
            pos = n->keys[i];
            if (pos)
            {
                new_node->keys[child] = i;
                art_node *child_ptr = n->children[pos - 1];
                new_node->children[child] = child_ptr;
#if SELF_REF
                if (!IS_LEAF(child_ptr))
                    child_ptr->self_ref = &new_node->children[child];
                else
#if LEAF_REF
                    LEAF_RAW(child_ptr)->self_ref = &new_node->children[child];
#endif
#endif
                child++;
            }
        }

        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node48_local_kind)
        {
            node48_local_cnt--;
        }
        else
        {
            node48_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node48_local_alloc_cnt, &node48_cxl_alloc_cnt, node48_hot, node48_cold);
#endif
#if CNT
        node48_cnt--;
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
#if SELF_REF
    refresh_self_refs((art_node *)n, pos, n->n.num_children); // recheck: -1 or not?
#endif
    n->n.num_children--;

    // Downgrade to node4 if number of children drops to 3
    if (n->n.num_children == 3)
    {
#if STATIC
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4, ((art_node *)n)->depth);
#else
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
#endif
        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n, ref);

        memcpy(new_node->keys, n->keys, 3); // only 3 keys remain
        memcpy(new_node->children, n->children, 3 * sizeof(void *));

#if SELF_REF
        for (int i = 0; i < 4; i++)
        {
            art_node *child = new_node->children[i];
            if (!IS_LEAF(child))
                child->self_ref = &new_node->children[i];
            else
            {
#if LEAF_REF
                LEAF_RAW(child)->self_ref = &new_node->children[i];
#endif
            }
        }
#endif
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node16_local_kind)
        {
            node16_local_cnt--;
        }
        else
        {
            node16_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node16_local_alloc_cnt, &node16_cxl_alloc_cnt, node16_hot, node16_cold);
#endif
#if CNT
        node16_cnt--;
#endif
    }
}

static void remove_child4(art_node4 *n, art_node **ref, art_node **l)
{
    int pos = l - n->children;
    memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - 1 - pos);
    memmove(n->children + pos, n->children + pos + 1, (n->n.num_children - 1 - pos) * sizeof(void *));
#if SELF_REF
    refresh_self_refs((art_node *)n, pos, n->n.num_children);
#endif
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
#if DEPTH_INDI
        child->depth = n->n.depth;
#endif
#if SELF_REF
        if (!IS_LEAF(child))
            child->self_ref = ref;
        else
        {
#if LEAF_REF
            LEAF_RAW(child)->self_ref = ref;
#endif
        }
#endif
        struct memkind *kind = memkind_detect_kind((void *)n);
        memkind_free(kind, n);
#if ENABLE_PROFILE
        if (kind == node4_local_kind)
        {
            node4_local_cnt--;
        }
        else
        {
            node4_cxl_cnt--;
        }
#endif
#if ONLINE
        // del_node_from_arr((art_node *)n, kind);
        del_node_from_arr((art_node *)n, &node4_local_alloc_cnt, &node4_cxl_alloc_cnt, node4_hot, node4_cold);
#endif
#if CNT
        node4_cnt--;
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
        // #if STREAM_ACC_ADDR
        //         if (start_acc_streaming)
        //             fprintf(acc_fd, "%lu,0\n", (unsigned long)n);
        // #endif
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
// #if CLFLUSH4 || CLFLUSH16 || CLFLUSH48 || CLFLUSH256
//     switch (n->type)
//     {
// #if CLFLUSH4
//     case NODE4:
//         clflush_range((void *)n, sizeof(art_node4));
//         break;
// #endif
// #if CLFLUSH16
//     case NODE16:
//         clflush_range((void *)n, sizeof(art_node16));
//         break;
// #endif
// #if CLFLUSH48
//     case NODE48:
//         clflush_range((void *)n, sizeof(art_node48));
//         break;
// #endif
// #if CLFLUSH256
//     case NODE256:
//         clflush_range((void *)n, sizeof(art_node256));
//         break;
// #endif
//     }
// #endif
#if HIT_DIST
    n->hit_cnt++;
#endif
    // #if STREAM_ACC_ADDR
    //     if (start_acc_streaming)
    //         fprintf(acc_fd, "%lu,%d\n", (unsigned long)n, type_map[n->type]);
    // #endif

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
        // struct memkind *kind = memkind_detect_kind((void *)l);
        // memkind_free(kind, l);
        free(l);
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
#if HIT_CNT_TOTAL
        leaf_hit_cnt++;
#endif
        art_leaf *l = LEAF_RAW(n);
        return cb(data, (const unsigned char *)l->key, l->key_len, l->value);
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
#if CLFLUSH4 || CLFLUSH16 || CLFLUSH48 || CLFLUSH256
    switch (n->type)
    {
#if CLFLUSH4
    case NODE4:
        clflush_range((void *)n, sizeof(art_node4));
        break;
#endif
#if CLFLUSH16
    case NODE16:
        clflush_range((void *)n, sizeof(art_node16));
        break;
#endif
#if CLFLUSH48
    case NODE48:
        clflush_range((void *)n, sizeof(art_node48));
        break;
#endif
#if CLFLUSH256
    case NODE256:
        clflush_range((void *)n, sizeof(art_node256));
        break;
#endif
    }
#endif
#if HIT_DIST
    n->hit_cnt++;
#endif
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

#if FIRST_TOUCH
#else
    unsigned long node_mask = 1UL << (use_cxl ? CXL_MASK : LOCAL_MASK);
    long mbind_ret = mbind(*base, size, MPOL_BIND, &node_mask, sizeof(node_mask) * 8, 0);
    if (mbind_ret != 0)
    {
        perror("mbind");
        abort();
    }
#endif
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
    float node4_mb = node4_cnt * NODE4_SIZE / (1024 * 1024);
    float node16_mb = node16_cnt * NODE16_SIZE / (1024 * 1024);
    float node48_mb = node48_cnt * NODE48_SIZE / (1024 * 1024);
    float node256_mb = node256_cnt * NODE256_SIZE / (1024 * 1024);
    float total_mb = node4_mb + node16_mb + node48_mb + node256_mb;

    printf("Node4: %lu, %.2f MB, %.2f pages\n", node4_cnt, node4_mb, ceil(node4_mb * 256));
    printf("Node16: %lu, %.2f MB, %.2f pages\n", node16_cnt, node16_mb, ceil(node16_mb * 256));
    printf("Node48: %lu, %.2f MB, %.2f pages\n", node48_cnt, node48_mb, ceil(node48_mb * 256));
    printf("Node256: %lu, %.2f MB, %.2f pages\n", node256_cnt, node256_mb, ceil(node256_mb * 256));
    printf("total: %.2f MB\n", total_mb);
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
#if DEPTH_INDI
void collect_node_depths(art_node *n, int depth, node_depth_stats_t *stats, FILE *fd)
{
    if (!n)
        return;
    if (IS_LEAF(n))
    {
        //         fprintf(fd, "leaf: %d", depth);
        //         art_leaf *leaf = LEAF_RAW(n);
        // #if DEPTH_INDI
        //         fprintf(fd, " %d", leaf->depth);
        // #endif
        //         fprintf(fd, " %s", leaf->key);
        //         fprintf(fd, " \n");
        return;
    }
    switch (n->type)
    {
    case NODE4:
    {
        fprintf(fd, "4: %p %d", n, depth);
        fprintf(fd, " %d", n->depth);
        fprintf(fd, " \n");
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
        fprintf(fd, "16: %p %d", n, depth);
        fprintf(fd, " %d", n->depth);
        fprintf(fd, " \n");
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
        fprintf(fd, "48: %p %d", n, depth);
        fprintf(fd, " %d", n->depth);
        fprintf(fd, " \n");
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
        fprintf(fd, "256: %p %d", n, depth);
        fprintf(fd, " %d", n->depth);
        fprintf(fd, " \n");
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

#if ONLINE
art_node *tiered_calloc(bool *local_full,
                        memkind_t local_kind, memkind_t cxl_kind,
                        size_t size,
                        art_node **node_hot_arr, int *hot_count,
                        art_node **node_cold_arr, int *cold_count)
// art_node *tiered_calloc(bool *local_full, struct memkind *local_kind, struct memkind *cxl_kind, size_t size, uint8_t type)
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
            char msg[256];
            memkind_error_message(MEMKIND_ERROR_MALLOC, msg, sizeof(msg));
            fprintf(stderr, "memkind_calloc() failed: %s\n", msg);
            return NULL;
        }
        node_cold_arr[*cold_count] = ptr;
        ptr->idx_in_arr = *cold_count;
        ptr->in_local = false;
        (*cold_count)++;
    }
    return ptr;
}

static int sort_ascending(const void *a, const void *b)
{
    const art_node *na = *(const art_node **)a;
    const art_node *nb = *(const art_node **)b;
    return na->hit_cnt - nb->hit_cnt;
}

static int sort_descending(const void *a, const void *b)
{
    const art_node *na = *(const art_node **)a;
    const art_node *nb = *(const art_node **)b;
    return nb->hit_cnt - na->hit_cnt;
}

static int sort_descending_r(const void *a, const void *b, void *arg)
{
    uint8_t type = *(uint8_t *)arg;
    const art_node *na = *(const art_node **)a;
    const art_node *nb = *(const art_node **)b;
    size_t density_a = 0;
    size_t density_b = 0;
    switch (type)
    {
    case NODE4:
        density_a = NODE4_SENSI * na->hit_cnt;
        density_b = NODE4_SENSI * nb->hit_cnt;
        break;
    case NODE16:
        density_a = NODE16_SENSI * na->hit_cnt;
        density_b = NODE16_SENSI * nb->hit_cnt;
        break;
    case NODE48:
        density_a = NODE48_SENSI * na->hit_cnt;
        density_b = NODE48_SENSI * nb->hit_cnt;
        break;
    case NODE256:
        density_a = NODE256_SENSI * na->hit_cnt;
        density_b = NODE256_SENSI * nb->hit_cnt;
        break;
    default:
        abort();
    }
    return density_b - density_a;
}
static int sort_ascending_r(const void *a, const void *b, void *arg)
{
    uint8_t type = *(uint8_t *)arg;
    const art_node *na = *(const art_node **)a;
    const art_node *nb = *(const art_node **)b;
    size_t density_a = 0;
    size_t density_b = 0;
    switch (type)
    {
    case NODE4:
        density_a = NODE4_SENSI * na->hit_cnt;
        density_b = NODE4_SENSI * nb->hit_cnt;
        break;
    case NODE16:
        density_a = NODE16_SENSI * na->hit_cnt;
        density_b = NODE16_SENSI * nb->hit_cnt;
        break;
    case NODE48:
        density_a = NODE48_SENSI * na->hit_cnt;
        density_b = NODE48_SENSI * nb->hit_cnt;
        break;
    case NODE256:
        density_a = NODE256_SENSI * na->hit_cnt;
        density_b = NODE256_SENSI * nb->hit_cnt;
        break;
    default:
        abort();
    }
    return density_a - density_b;
}
static void sort_hotness(art_node **alloced_nodes, int alloced_cnt, bool acsending)
{ // not used
    if (alloced_cnt < 1000)
        return;
    printf("sorting total %d\n", alloced_cnt);
    if (acsending)
    {
        // qsort_r(alloced_nodes, alloced_cnt, sizeof(void *), sort_ascending_r, &type);
        qsort(alloced_nodes, alloced_cnt, sizeof(void *), sort_ascending); // too slow, 5M/s
    }
    else
    {
        // qsort_r(alloced_nodes, alloced_cnt, sizeof(void *), sort_descending_r, &type);
        qsort(alloced_nodes, alloced_cnt, sizeof(void *), sort_descending);
    }

    // printf("sort result: \n");
    // for (int i = 0; i < alloced_cnt; i++)
    // {
    //     printf("node[%d] = %p, hit_cnt = %d, self_ref = %d\n",
    //            i,
    //            alloced_nodes[i],
    //            ((art_node *)alloced_nodes[i])->hit_cnt,
    //            ((art_node *)alloced_nodes[i])->self_ref);
    // }
}
void heapify(art_node **arr, int N, int i)
{
    int smallest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < N && arr[l]->hit_cnt < arr[smallest]->hit_cnt)
        smallest = l;
    if (r < N && arr[r]->hit_cnt < arr[smallest]->hit_cnt)
        smallest = r;

    if (smallest != i)
    {
        art_node *temp = arr[i];
        arr[i] = arr[smallest];
        arr[smallest] = temp;
        heapify(arr, N, smallest);
    }
}

// Build a min-heap
void build_min_heap(art_node **arr, int N)
{
    if (!arr || N <= 0)
        return;

    for (int i = N / 2 - 1; i >= 0; i--)
        heapify(arr, N, i);
}

static void swap_hot_cold_nodes_bulk(art_node **hot_node_arr, int hot_node_count, art_node **cold_node_arr, int cold_node_count)
{
    if (hot_node_count == 0 || cold_node_count == 0)
        return;

    size_t node_size = 0;
    switch (hot_node_arr[0]->type)
    {
    case NODE4:
        node_size = sizeof(art_node4);
        break;
    case NODE16:
        node_size = sizeof(art_node16);
        break;
    case NODE48:
        node_size = sizeof(art_node48);
        break;
    case NODE256:
        node_size = sizeof(art_node256);
        break;
    default:
        abort();
    }

    int max_possible_swaps = hot_node_count < cold_node_count ? hot_node_count : cold_node_count;
    max_possible_swaps = max_possible_swaps < TOP_K_SWAP ? max_possible_swaps : TOP_K_SWAP;

    // Preallocate a buffer to hold up to max_possible_swaps worth of node copies
    void *tmp_buf = calloc(max_possible_swaps, node_size);
    assert(tmp_buf);

    int cur_idx = 0;
    for (; cur_idx < max_possible_swaps; ++cur_idx)
    {
        art_node *hot_node = hot_node_arr[cur_idx];
        art_node *cold_node = cold_node_arr[cur_idx];

        if (hot_node->hit_cnt >= cold_node->hit_cnt)
            break;

        // Use the appropriate offset inside the tmp_buf
        void *tmp = (char *)tmp_buf + cur_idx * node_size;

        memcpy(tmp, hot_node, node_size);
        memcpy(hot_node, cold_node, node_size);
        memcpy(cold_node, tmp, node_size);

        if (hot_node->self_ref)
            *(hot_node->self_ref) = hot_node;
        if (cold_node->self_ref)
            *(cold_node->self_ref) = cold_node;
    }

    free(tmp_buf);
    printf("successfully swapped %d\n", cur_idx);
}
static void swap_hot_cold_nodes(art_node **hot_node_arr, int hot_node_count, art_node **cold_node_arr, int cold_node_count)
{
    int cur_idx = 0;
    size_t node_size = 0;
    switch (hot_node_arr[0]->type)
    {
    case NODE4:
        node_size = sizeof(art_node4);
        break;
    case NODE16:
        node_size = sizeof(art_node16);
        break;
    case NODE48:
        node_size = sizeof(art_node48);
        break;
    case NODE256:
        node_size = sizeof(art_node256);
        break;
    default:
        abort();
    }
    while (cur_idx < hot_node_count && cur_idx < cold_node_count && cur_idx < TOP_K_SWAP)
    {
        art_node *hot_node = hot_node_arr[cur_idx];
        art_node *cold_node = cold_node_arr[cur_idx];
        if (hot_node->hit_cnt >= cold_node->hit_cnt) // when local's cold nodes are not really colder than CXL's hot nodes, stop
            break;

        void *tmp = calloc(1, node_size);
        assert(tmp);
        memcpy(tmp, hot_node, node_size);
        memcpy(hot_node, cold_node, node_size);
        memcpy(cold_node, tmp, node_size);
        free(tmp);

        if (hot_node->self_ref)
            *(hot_node->self_ref) = hot_node;
        if (cold_node->self_ref)
            *(cold_node->self_ref) = cold_node;

        cur_idx++;
    }
    printf("successfully swapped %d\n", cur_idx);
}
void partial_sort_top_k(art_node **arr, int N, int K, bool ascending)
{
    // struct timespec build_heap_start, build_heap_end, heapify_start, heapify_end, qsort_start, qsort_end;
    // double build_heap_ms, heapify_ms, qsort_ms = 0;
    if (!arr || N <= 0 || K <= 0)
        return;
    if (K > N)
        K = N;

    // Step 1: Make a copy of arr to heapify
    // art_node **heap = malloc(sizeof(art_node *) * N);
    // if (!heap)
    //     return;

    // for (int i = 0; i < N; ++i)
    //     heap[i] = arr[i];

    // clock_gettime(CLOCK_MONOTONIC, &build_heap_start);
    // build_min_heap(heap, N); // heap is now a valid min-heap of size N
    // clock_gettime(CLOCK_MONOTONIC, &build_heap_end);
    // build_heap_ms = elapsed_ms(build_heap_start, build_heap_end);

    // Step 2: Extract top K into a sorted array
    // clock_gettime(CLOCK_MONOTONIC, &heapify_start);
    // art_node *sorted[K];
    // int heap_size = N;
    // for (int i = 0; i < K; ++i)
    // {
    //     sorted[i] = heap[0]; // take min
    //     heap[0] = heap[--heap_size];
    //     heapify(heap, heap_size, 0);
    // }
    // clock_gettime(CLOCK_MONOTONIC, &heapify_end);
    // heapify_ms = elapsed_ms(heapify_start, heapify_end);

    // Step 3: Final sort
    // clock_gettime(CLOCK_MONOTONIC, &qsort_start);
    qsort(arr, N, sizeof(art_node *),
          ascending ? sort_ascending : sort_descending);
    // clock_gettime(CLOCK_MONOTONIC, &qsort_end);
    // qsort_ms = elapsed_ms(qsort_start, qsort_end);

    for (int i = 0; i < K; i++)
    {
        fprintf(log_fd, "%d\n", arr[i]->hit_cnt);
    }
    // printf("build_heap: %.3f, heapify: %.3f, qsort: %.3f\n", build_heap_ms / 1000, heapify_ms / 1000, qsort_ms / 1000);

    // free(heap);
}
void sort_all_hotness()
{
    // partial_sort_top_k(node4_hot, node4_local_alloc_cnt, TOP_K_SWAP, true); // sort everything with high overhead, do not use
    // partial_sort_top_k(node4_cold, node4_cxl_alloc_cnt, TOP_K_SWAP, false);
    // partial_sort_top_k(node16_hot, node16_local_alloc_cnt, TOP_K_SWAP, true);
    // partial_sort_top_k(node16_cold, node16_cxl_alloc_cnt, TOP_K_SWAP, false);
    // partial_sort_top_k(node48_hot, node48_local_alloc_cnt, TOP_K_SWAP, true);
    // partial_sort_top_k(node48_cold, node48_cxl_alloc_cnt, TOP_K_SWAP, false);
    // partial_sort_top_k(node256_hot, node256_local_alloc_cnt, TOP_K_SWAP, true);
    // partial_sort_top_k(node256_cold, node256_cxl_alloc_cnt, TOP_K_SWAP, false);

    art_node *cold_node4_in_local[TOP_K_SWAP];
    int cold_node4_cnt = 0;
    get_top_k_by_hit_cnt(cold_node4_in_local, &cold_node4_cnt, node4_hot, node4_local_alloc_cnt, TOP_K_SWAP, true);
    // for debugging
    // for (int i = 0; i < cold_node4_cnt; i++)
    // {
    //     fprintf(log_fd, "%d\n", cold_node4_in_local[i]->hit_cnt);
    // }
    art_node *hot_node4_in_cxl[TOP_K_SWAP];
    int hot_node4_cnt = 0;
    get_top_k_by_hit_cnt(hot_node4_in_cxl, &hot_node4_cnt, node4_cold, node4_cxl_alloc_cnt, TOP_K_SWAP, false);
    // swap_hot_cold_nodes(cold_node4_in_local, cold_node4_cnt, hot_node4_in_cxl, hot_node4_cnt);
    swap_hot_cold_nodes_bulk(cold_node4_in_local, cold_node4_cnt, hot_node4_in_cxl, hot_node4_cnt);

    art_node *cold_node16_in_local[TOP_K_SWAP];
    int cold_node16_cnt = 0;
    get_top_k_by_hit_cnt(cold_node16_in_local, &cold_node16_cnt, node16_hot, node16_local_alloc_cnt, TOP_K_SWAP, true);
    art_node *hot_node16_in_cxl[TOP_K_SWAP];
    int hot_node16_cnt = 0;
    get_top_k_by_hit_cnt(hot_node16_in_cxl, &hot_node16_cnt, node16_cold, node16_cxl_alloc_cnt, TOP_K_SWAP, false);

    art_node *cold_node48_in_local[TOP_K_SWAP];
    int cold_node48_cnt = 0;
    get_top_k_by_hit_cnt(cold_node48_in_local, &cold_node48_cnt, node48_hot, node48_local_alloc_cnt, TOP_K_SWAP, true);
    art_node *hot_node48_in_cxl[TOP_K_SWAP];
    int hot_node48_cnt = 0;
    get_top_k_by_hit_cnt(hot_node48_in_cxl, &hot_node48_cnt, node48_cold, node48_cxl_alloc_cnt, TOP_K_SWAP, false);

    art_node *cold_node256_in_local[TOP_K_SWAP];
    int cold_node256_cnt = 0;
    get_top_k_by_hit_cnt(cold_node256_in_local, &cold_node256_cnt, node256_hot, node256_local_alloc_cnt, TOP_K_SWAP, true);
    art_node *hot_node256_in_cxl[TOP_K_SWAP];
    int hot_node256_cnt = 0;
    get_top_k_by_hit_cnt(hot_node256_in_cxl, &hot_node256_cnt, node256_cold, node256_cxl_alloc_cnt, TOP_K_SWAP, false);
}
// art_node * n: node to remove
// int *local_alloc_cnt:  current number of nodes in local (hot) array
// int *cxl_alloc_cnt: current number of nodes in cold array
// art_node **hot_arr: array of local allocations
// art_node **cold_arr: array of CXL allocations
// Explanation: “swap-and-pop” removal from a dynamic array.
// 	1.	Removes node n from either hot_arr or cold_arr.
// 	2.	Maintains compactness of the array by swapping the last element into the removed spot.
// 	3.	Updates idx_in_arr on the moved element.
// 	4.	Marks the removed node’s idx_in_arr as -1 (invalid).
static void del_node_from_arr(art_node *n, int *local_alloc_cnt, int *cxl_alloc_cnt, art_node **hot_arr, art_node **cold_arr)
// static void del_node_from_arr(art_node *n, struct memkind *kind)
{
    return;
    int idx = n->idx_in_arr;
    int last = 0;
    n->idx_in_arr = -1; // Mark as no longer tracked
    if (n->in_local)
    {
        last = *local_alloc_cnt - 1;
        if (idx != last)
        {
            hot_arr[idx] = hot_arr[last];                 // move last node into deleted slot
            ((art_node *)hot_arr[idx])->idx_in_arr = idx; // fix moved node's idx
        }
        (*local_alloc_cnt)--; // shrink array
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
    // del_node_from_set(n, kind);
}

// don't use, too heavy and delaying tracing
void traverse_tree_populate_min_heap(art_node *n)
{
    if (!n || IS_LEAF(n))
        return;
    traverse_cnt++;
    // if (n->type != NODE4)
    // {
    //     populate_min_heap(n);
    // }

    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            traverse_tree_populate_min_heap(node->children[i]);
        break;
    }
    case NODE16:
    {
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
            traverse_tree_populate_min_heap(node->children[i]);
        break;
    }
    case NODE48:
    {
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->keys[i])
            {
                int idx = node->keys[i] - 1;
                traverse_tree_populate_min_heap(node->children[idx]);
            }
        }
        break;
    }
    case NODE256:
    {
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
                traverse_tree_populate_min_heap(node->children[i]);
        }
        break;
    }
    default:
        break;
    }
}
#endif

#if DFS
void dfs_print_hit_cnt_path(art_node *node, int depth, int *path, void **node_path, FILE *fd)
{
    if (!node)
        return;

    if (IS_LEAF(node))
    {
        // Print path so far
        int total_hit = 0;
        for (int i = 0; i < depth; i++)
        {
            if (i == 0)
                continue;
            total_hit += path[i];
            fprintf(fd, "%p,%d,", node_path[i], path[i]); // node addr and hit_cnt
            // fprintf(fd, "%d,", path[i]); // only hit_cnt
        }
        fprintf(fd, "%d\n", total_hit);
        // fprintf(fd, "%d,LEAF:%p\n", total_hit, node);
        return;
    }
    node_path[depth] = (void *)node;
    path[depth] = node->hit_cnt;
    depth++;

    switch (node->type)
    {
    case NODE4:
    {
        art_node4 *n = (art_node4 *)node;
        for (int i = 0; i < n->n.num_children; i++)
        {
            dfs_print_hit_cnt_path(n->children[i], depth, path, node_path, fd);
        }
        break;
    }
    case NODE16:
    {
        art_node16 *n = (art_node16 *)node;
        for (int i = 0; i < n->n.num_children; i++)
        {
            dfs_print_hit_cnt_path(n->children[i], depth, path, node_path, fd);
        }
        break;
    }
    case NODE48:
    {
        art_node48 *n = (art_node48 *)node;
        for (int i = 0; i < 256; i++)
        {
            int pos = n->keys[i];
            if (pos)
                dfs_print_hit_cnt_path(n->children[pos - 1], depth, path, node_path, fd);
        }
        break;
    }
    case NODE256:
    {
        art_node256 *n = (art_node256 *)node;
        for (int i = 0; i < 256; i++)
        {
            if (n->children[i])
                dfs_print_hit_cnt_path(n->children[i], depth, path, node_path, fd);
        }
        break;
    }
    default:
        break;
    }
}
#endif

static void print_indent(FILE *out, int indent)
{
    for (int i = 0; i < indent; i++)
    {
        fprintf(out, "  ");
    }
}

#if SELF_REF
void dump_self_ref_json(FILE *out, art_node *n, void *parent_child_ptr)
{
    if (!n)
    {
        fprintf(out, "null");
        return;
    }

    if (IS_LEAF(n))

    {

        // You can extend this to emit leaf-specific info if needed

        fprintf(out, "{ \"addr\": \"%p\", \"type\": \"leaf\"", (void *)n);

#if LEAF_REF
        // fprintf(out, ", \"parent_child_ptr\": \"%p\"\n", parent_child_ptr);
        fprintf(out, ", \"parent_child_ptr\": \"%p\"\n", parent_child_ptr);
        fprintf(out, ", \"parent\": \"%p\"}\n", LEAF_RAW(n)->self_ref);
#else

        fprintf(out, " }\n");

#endif

        return;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"parent_child_ptr\": \"%p\",\n", parent_child_ptr);
    fprintf(out, "  \"addr\": \"%p\",\n", (void *)n);
    fprintf(out, "  \"type\": %d,\n", n->type);
    fprintf(out, "  \"parent\": \"%p\",\n", (void *)n->self_ref);
    fprintf(out, "  \"children\": [\n");

    bool first = true;
    switch (n->type)
    {
    case NODE4:
    {
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
        {
            if (!first)
                fprintf(out, ",\n");
            first = false;
            dump_self_ref_json(out, node->children[i], &node->children[i]);
        }
        break;
    }
    case NODE16:
    {
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
        {
            if (!first)
                fprintf(out, ",\n");
            first = false;
            dump_self_ref_json(out, node->children[i], &node->children[i]);
        }
        break;
    }
    case NODE48:
    {
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->keys[i])
            {
                int idx = node->keys[i] - 1;
                if (!first)
                    fprintf(out, ",\n");
                first = false;
                dump_self_ref_json(out, node->children[idx], &node->children[idx]);
            }
        }
        break;
    }
    case NODE256:
    {
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
            {
                if (!first)
                    fprintf(out, ",\n");
                first = false;
                dump_self_ref_json(out, node->children[i], &node->children[i]);
            }
        }
        break;
    }
    default:
        fprintf(out, "    { \"error\": \"Unknown node type %d\" }", n->type);
        break;
    }

    fprintf(out, "\n  ]\n");
    fprintf(out, "}");
}
#endif

#if STATIC
#define COLS 4 // number of columns (node types: 4, 16, 48, 256)
#define MAX_LINE_LEN 128
static void load_static_metrics(char *wl)
{
    char filename[64];
    snprintf(filename, sizeof(filename),
             "static_placement_%s.txt", wl);
    // const char *filename = "static_placement_email_workloadc_ext_10.txt";
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open matrix file");
        exit(1);
    }

    // First, count number of lines (rows)
    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), file))
    {
        if (strlen(line) > 1) // skip empty lines
            static_metrics_line_cnt++;
    }

    matrix = malloc(static_metrics_line_cnt * sizeof(int *));
    if (!matrix)
    {
        perror("Memory allocation failed");
        fclose(file);
        return;
    }
    for (int i = 0; i < static_metrics_line_cnt; i++)
    {
        matrix[i] = calloc(COLS, sizeof(int));
        if (!matrix[i])
        {
            perror("Memory allocation failed");
            fclose(file);
            return;
        }
    }

    // Rewind file and parse content into matrix
    rewind(file);
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < static_metrics_line_cnt)
    {
        int a, b, c, d;
        if (sscanf(line, "%d %d %d %d", &a, &b, &c, &d) == 4)
        {
            matrix[row][0] = a;
            matrix[row][1] = b;
            matrix[row][2] = c;
            matrix[row][3] = d;
            row++;
        }
    }

    fclose(file);
    // printf("Parsed matrix (%d rows, %d cols):\n", static_metrics_line_cnt, COLS);
    // for (int i = 0; i < static_metrics_line_cnt; i++)
    // {
    //     for (int j = 0; j < COLS; j++)
    //     {
    //         printf("%d ", matrix[i][j]);
    //     }
    //     printf("\n");
    // }
}
static void free_static_metrics()
{
    // Free memory
    for (int i = 0; i < static_metrics_line_cnt; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
    matrix = NULL;
}

#endif