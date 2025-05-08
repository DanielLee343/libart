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
#define INIT_LEAF_COUNT 100000
#define NUM_LEAF_PAGE 912316
#define NUM_256_PAGE 1024
#define NUM_48_PAGE 98750
#define NUM_16_PAGE 210938
#define NUM_4_PAGE 387500

#define LEAF_CXL 0
#define NODE4_CXL 0
#define NODE16_CXL 0
#define NODE48_CXL 0
#define NODE256_CXL 0

#define CXL_MASK 1
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
#if NODE4_CXL
        n = ALIGN_UP(node4_ptr, LEAF_ALIGN);
        node4_ptr = (void *)((uintptr_t)n + sizeof(art_node4));
#else
        n = (art_node *)calloc(1, sizeof(art_node4));
#endif
        node4_cnt++;
        break;
    case NODE16:
#if NODE16_CXL
        n = ALIGN_UP(node16_ptr, LEAF_ALIGN);
        node16_ptr = (void *)((uintptr_t)n + sizeof(art_node16));
#else
        n = (art_node *)calloc(1, sizeof(art_node16));
#endif
        node16_cnt++;
        break;
    case NODE48:
#if NODE48_CXL
        n = ALIGN_UP(node48_ptr, LEAF_ALIGN);
        node48_ptr = (void *)((uintptr_t)n + sizeof(art_node48));
#else
        n = (art_node *)calloc(1, sizeof(art_node48));
#endif
        node48_cnt++;
        break;
    case NODE256:
#if NODE256_CXL
        n = ALIGN_UP(node256_ptr, LEAF_ALIGN);
        node256_ptr = (void *)((uintptr_t)n + sizeof(art_node256));
#else
        n = (art_node *)calloc(1, sizeof(art_node256));
#endif
        node256_cnt++;
        break;
    default:
        abort();
    }
    n->type = type;
    return n;
}

/**
 * Initializes an ART tree
 * @return 0 on success.
 */
int art_tree_init(art_tree *t)
{
    t->root = NULL;
    t->size = 0;
#if LEAF_CXL
    // slab_base = numa_alloc_onnode((size_t)PAGE_SIZE * INIT_LEAF_COUNT, CXL_MASK);
    // memset(slab_base, 0, (size_t)PAGE_SIZE * INIT_LEAF_COUNT);
    // total_leaf_count = INIT_LEAF_COUNT;
    slab_base = numa_alloc_onnode((size_t)PAGE_SIZE * NUM_LEAF_PAGE, CXL_MASK);
    memset(slab_base, 0, (size_t)PAGE_SIZE * NUM_LEAF_PAGE);
    bump_ptr = slab_base;
#else
    size_t total_leaf_size = (size_t)PAGE_SIZE * NUM_LEAF_PAGE;
    leaf_base = mmap(NULL, total_leaf_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(leaf_base != NULL);
    unsigned long local_mask = 1UL << 1; // local DRAM
    long mbind_ret = mbind(leaf_base, total_leaf_size, MPOL_BIND, &local_mask, sizeof(local_mask) * 2, 0);
    if (mbind_ret != 0)
    {
        perror("mbind");
        abort();
    }
    int err = memkind_create_fixed(leaf_base, total_leaf_size, &leaf_kind);
    if (err)
    {
        perror("memkind_create_fixed failed");
        abort();
    }
#endif
#if NODE256_CXL
    node256_base = numa_alloc_onnode((size_t)PAGE_SIZE * NUM_256_PAGE, CXL_MASK);
    memset(node256_base, 0, (size_t)PAGE_SIZE * NUM_256_PAGE);
    node256_ptr = node256_base;
#endif
#if NODE48_CXL
    node48_base = numa_alloc_onnode((size_t)PAGE_SIZE * NUM_48_PAGE, CXL_MASK);
    memset(node48_base, 0, (size_t)PAGE_SIZE * NUM_48_PAGE);
    node48_ptr = node48_base;
#endif
#if NODE16_CXL
    node16_base = numa_alloc_onnode((size_t)PAGE_SIZE * NUM_16_PAGE, CXL_MASK);
    memset(node16_base, 0, (size_t)PAGE_SIZE * NUM_16_PAGE);
    node16_ptr = node16_base;
#endif
#if NODE4_CXL
    node4_base = numa_alloc_onnode((size_t)PAGE_SIZE * NUM_4_PAGE, CXL_MASK);
    memset(node4_base, 0, (size_t)PAGE_SIZE * NUM_4_PAGE);
    node4_ptr = node4_base;
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
        leaf_cnt--;
#if LEAF_CXL
        // fprintf(stdout, "Destroying leaf at %p (leaf_cnt now %lu)\n", LEAF_RAW(n), leaf_cnt);
        // art_leaf *leaf = LEAF_RAW(n);
        // numa_free(leaf, sizeof(art_leaf) + leaf->key_len);
#else
        // free(LEAF_RAW(n)); // vanilla
        memkind_free(leaf_kind, LEAF_RAW(n));
#endif
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
    switch (n->type)
    {
    case NODE4:
        p.p1 = (art_node4 *)n;
        for (i = 0; i < n->num_children; i++)
        {
            destroy_node(p.p1->children[i]);
        }
#if NODE4_CXL
        // numa_free(n, sizeof(art_node4));
#else
        free(n);
#endif
        node4_cnt--;
        break;

    case NODE16:
        p.p2 = (art_node16 *)n;
        for (i = 0; i < n->num_children; i++)
        {
            destroy_node(p.p2->children[i]);
        }
#if NODE16_CXL
        // numa_free(n, sizeof(art_node16));
#else
        free(n);
#endif
        node16_cnt--;
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
#if NODE48_CXL
        // numa_free(n, sizeof(art_node48));
#else
        free(n);
#endif
        node48_cnt--;
        break;

    case NODE256:
        p.p4 = (art_node256 *)n;
        for (i = 0; i < 256; i++)
        {
            if (p.p4->children[i])
                destroy_node(p.p4->children[i]);
        }
#if NODE256_CXL
        // numa_free(n, sizeof(art_node256)); // do not do free here since we're using bulk allocation
#else
        free(n);
#endif
        node256_cnt--;
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
    // printf("num_unmatch: %zu\n", num_unmatch);
    destroy_node(t->root);
#if LEAF_CXL
    numa_free(slab_base, (size_t)PAGE_SIZE * NUM_LEAF_PAGE);
#else
    int err = memkind_destroy_kind(leaf_kind);
    if (err)
    {
        perror("memkind_destroy_kind failed");
        abort();
    }
    munmap(leaf_base, total_leaf_size);
    fprintf(stdout, "Memory was successfully allocated and released.\n");
#endif
#if NODE256_CXL
    numa_free(node256_base, (size_t)PAGE_SIZE * NUM_256_PAGE);
#endif
#if NODE48_CXL
    numa_free(node48_base, (size_t)PAGE_SIZE * NUM_48_PAGE);
#endif
#if NODE16_CXL
    numa_free(node16_base, (size_t)PAGE_SIZE * NUM_16_PAGE);
#endif
#if NODE4_CXL
    numa_free(node4_base, (size_t)PAGE_SIZE * NUM_4_PAGE);
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
            n = (art_node *)LEAF_RAW(n);
            // Check if the expanded path matches
            if (!leaf_matches((art_leaf *)n, key, key_len, depth))
            {
                return ((art_leaf *)n)->value;
            }
            return NULL;
        }

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
#if LEAF_CXL
    l = ALIGN_UP(bump_ptr, LEAF_ALIGN);
    bump_ptr = (void *)((uintptr_t)l + sizeof(art_leaf) + key_len);
#else
    // l = (art_leaf *)calloc(1, sizeof(art_leaf) + key_len); // vanilla
    void *mem = memkind_calloc(leaf_kind, 1, sizeof(art_leaf) + key_len); // compare addr pattern
    l = (art_leaf *)mem;
#endif
    // printf("0x%lx, size: %lu\n", (unsigned long)(uintptr_t)l, sizeof(art_leaf) + key_len);
    l->value = value;
    l->key_len = key_len;
    memcpy(l->key, key, key_len);
    leaf_cnt++;
    return l;
}

// reallocation doens't work
// static art_leaf *make_leaf(const unsigned char *key, int key_len, void *value)
// {
// #if LEAF_CXL
//     // First align the bump pointer — don't write to it yet
//     void *aligned_ptr = ALIGN_UP(bump_ptr, LEAF_ALIGN);
//     size_t needed = (uintptr_t)aligned_ptr + sizeof(art_leaf) + key_len - (uintptr_t)slab_base;

//     if (needed > PAGE_SIZE * total_leaf_count)
//     {
//         size_t old_cap = total_leaf_count;
//         size_t old_size = PAGE_SIZE * old_cap;
//         size_t new_cap = old_cap * 2;
//         size_t new_size = PAGE_SIZE * new_cap;

//         uintptr_t used_offset = (uintptr_t)aligned_ptr - (uintptr_t)slab_base;

//         void *new_base = numa_realloc(slab_base, old_size, new_size);
//         if (!new_base)
//         {
//             perror("numa_realloc failed");
//             exit(EXIT_FAILURE);
//         }

//         slab_base = new_base;
//         total_leaf_count = new_cap;

//         // Realign again after slab_base moved
//         aligned_ptr = ALIGN_UP((void *)((uintptr_t)slab_base + used_offset), LEAF_ALIGN);
//     }

//     art_leaf *l = (art_leaf *)aligned_ptr;
//     bump_ptr = (void *)((uintptr_t)l + sizeof(art_leaf) + key_len);

//     assert(((uintptr_t)l % LEAF_ALIGN) == 0);
//     assert((uintptr_t)bump_ptr <= (uintptr_t)slab_base + PAGE_SIZE * total_leaf_count);
// #else
//     art_leaf *l = (art_leaf *)calloc(1, sizeof(art_leaf) + key_len);
// #endif

//     l->value = value;
//     l->key_len = key_len;
//     memcpy(l->key, key, key_len);
//     leaf_cnt++;
//     return l;
// }

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
        new_node->depth = ((art_node48 *)n)->depth;
        for (int i = 0; i < 256; i++)
        {
            if (n->keys[i])
            {
                new_node->children[i] = n->children[n->keys[i] - 1];
            }
        }
        copy_header((art_node *)new_node, (art_node *)n);
        *ref = (art_node *)new_node;
#if NODE48_CXL
        // numa_free(n, sizeof(art_node48));
#else
        free(n);
#endif
        node48_cnt--;
        add_child256(new_node, ref, c, child);
    }
}

static void add_child16(art_node16 *n, art_node **ref, unsigned char c, void *child)
{
    if (n->n.num_children < 16)
    {
        unsigned mask = (1 << n->n.num_children) - 1;

// support non-x86 architectures
#ifdef __i386__
        __m128i cmp;

        // Compare the key to all 16 stored keys
        cmp = _mm_cmplt_epi8(_mm_set1_epi8(c),
                             _mm_loadu_si128((__m128i *)n->keys));

        // Use a mask to ignore children that don't exist
        unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
#ifdef __amd64__
        __m128i cmp;

        // Compare the key to all 16 stored keys
        cmp = _mm_cmplt_epi8(_mm_set1_epi8(c),
                             _mm_loadu_si128((__m128i *)n->keys));

        // Use a mask to ignore children that don't exist
        unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
        // Compare the key to all 16 stored keys
        unsigned bitfield = 0;
        for (short i = 0; i < 16; ++i)
        {
            if (c < n->keys[i])
                bitfield |= (1 << i);
        }

        // Use a mask to ignore children that don't exist
        bitfield &= mask;
#endif
#endif

        // Check if less than any
        unsigned idx;
        if (bitfield)
        {
            idx = __builtin_ctz(bitfield);
            memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
            memmove(n->children + idx + 1, n->children + idx,
                    (n->n.num_children - idx) * sizeof(void *));
        }
        else
            idx = n->n.num_children;

        // Set the child
        n->keys[idx] = c;
        n->children[idx] = (art_node *)child;
        n->n.num_children++;
    }
    else
    {
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48);
        new_node->depth = ((art_node16 *)n)->depth;

        // Copy the child pointers and populate the key map
        memcpy(new_node->children, n->children,
               sizeof(void *) * n->n.num_children);
        for (int i = 0; i < n->n.num_children; i++)
        {
            new_node->keys[n->keys[i]] = i + 1;
        }
        copy_header((art_node *)new_node, (art_node *)n);
        *ref = (art_node *)new_node;
#if NODE16_CXL
        // numa_free(n, sizeof(art_node16));
#else
        free(n);
#endif
        node16_cnt--;
        add_child48(new_node, ref, c, child);
    }
}

static void add_child4(art_node4 *n, art_node **ref, unsigned char c, void *child)
{
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
        n->children[idx] = (art_node *)child;
        n->n.num_children++;
    }
    else
    {
        art_node16 *new_node = (art_node16 *)alloc_node(NODE16);
        new_node->depth = ((art_node4 *)n)->depth;

        // Copy the child pointers and the key map
        memcpy(new_node->children, n->children,
               sizeof(void *) * n->n.num_children);
        memcpy(new_node->keys, n->keys,
               sizeof(unsigned char) * n->n.num_children);
        copy_header((art_node *)new_node, (art_node *)n);
        *ref = (art_node *)new_node;
#if NODE4_CXL
        // numa_free(n, sizeof(art_node4));
#else
        free(n);
#endif
        node4_cnt--;
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
        new_node->depth = depth;

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
        new_node->depth = depth;
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

    // Resize to a node48 on underflow, not immediately to prevent
    // trashing if we sit on the 48/49 boundary
    if (n->n.num_children == 37)
    {
        art_node48 *new_node = (art_node48 *)alloc_node(NODE48);
        new_node->depth = ((art_node256 *)n)->depth;
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
#if NODE256_CXL
        // numa_free(n, sizeof(art_node256)); // do not do it here
#else
        free(n);
#endif
        node256_cnt--;
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
        new_node->depth = ((art_node48 *)n)->depth;
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
#if NODE48_CXL
        // numa_free(n, sizeof(art_node48));
#else
        free(n);
#endif
        node48_cnt--;
    }
}

static void remove_child16(art_node16 *n, art_node **ref, art_node **l)
{
    int pos = l - n->children;
    memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - 1 - pos);
    memmove(n->children + pos, n->children + pos + 1, (n->n.num_children - 1 - pos) * sizeof(void *));
    n->n.num_children--;

    if (n->n.num_children == 3)
    {
        art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
        new_node->depth = ((art_node16 *)n)->depth;
        *ref = (art_node *)new_node;
        copy_header((art_node *)new_node, (art_node *)n);
        memcpy(new_node->keys, n->keys, 4);
        memcpy(new_node->children, n->children, 4 * sizeof(void *));
#if NODE16_CXL
        // numa_free(n, sizeof(art_node16));
#else
        free(n);
#endif
        node16_cnt--;
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
#if NODE4_CXL
        // numa_free(n, sizeof(art_node4));
#else
        free(n);
#endif
        node4_cnt--;
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
        art_leaf *l = LEAF_RAW(n);
        if (!leaf_matches(l, key, key_len, depth))
        {
            *ref = NULL;
            return l;
        }
        return NULL;
    }

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
        leaf_cnt--;
#if LEAF_CXL
        // numa_free(l, sizeof(art_leaf) + l->key_len);
#else
        free(l);
#endif
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

void node_cnt_stat()
{
    printf("------- Leaf Node Count Summry -------\n");
    printf("Node4: %lu\n", node4_cnt);
    printf("Node16: %lu\n", node16_cnt);
    printf("Node48: %lu\n", node48_cnt);
    printf("Node256: %lu\n", node256_cnt);
    printf("Leaf: %lu\n", leaf_cnt);
}

void collect_node_depths(art_node *n, int depth, node_depth_stats_t *stats)
{
    if (!n)
        return;
    if (IS_LEAF(n))
        return;

    switch (n->type)
    {
    case NODE4:
    {
        stats->node4_depth_total += depth;
        stats->node4_count++;
        art_node4 *node = (art_node4 *)n;
        for (int i = 0; i < node->n.num_children; i++)
        {
            collect_node_depths(node->children[i], depth + 1, stats);
        }
        break;
    }

    case NODE16:
    {
        stats->node16_depth_total += depth;
        stats->node16_count++;
        art_node16 *node = (art_node16 *)n;
        for (int i = 0; i < node->n.num_children; i++)
        {
            collect_node_depths(node->children[i], depth + 1, stats);
        }
        break;
    }

    case NODE48:
    {
        stats->node48_depth_total += depth;
        stats->node48_count++;
        art_node48 *node = (art_node48 *)n;
        for (int i = 0; i < 256; i++)
        {
            uint8_t idx = node->keys[i];
            if (idx)
            {
                collect_node_depths(node->children[idx - 1], depth + 1, stats);
            }
        }
        break;
    }

    case NODE256:
    {
        stats->node256_depth_total += depth;
        stats->node256_count++;
        art_node256 *node = (art_node256 *)n;
        for (int i = 0; i < 256; i++)
        {
            if (node->children[i])
            {
                collect_node_depths(node->children[i], depth + 1, stats);
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

int check_numa_node(void *addr)
{
    int status;
    int ret;

    // Check the NUMA node of the page that contains addr
    ret = move_pages(0,       // self process
                     1,       // one page
                     &addr,   // address
                     NULL,    // don't move
                     &status, // output node status
                     0);      // flags

    if (ret == -1)
    {
        perror("check addr loc failed");
        return -1;
    }

    if (status >= 0)
    {
        // printf("Address %p is on NUMA node %d\n", addr, status);
        return status;
    }
    else
    {
        // printf("Address %p is not currently mapped (status = %d)\n", addr, status);
        return -1;
    }
}
