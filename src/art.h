#include <stdint.h>
#include <stddef.h>
#include <numa.h>
#include <numaif.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <memkind.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#ifndef ART_H
#define ART_H

#ifdef __cplusplus
extern "C"
{
#endif

#define NODE4 1
#define NODE16 2
#define NODE48 3
#define NODE256 4

#define TOP_K_SWAP 10000
#define HOT_CACHE_LIMIT 10 * TOP_K_SWAP

#define MAX_PREFIX_LEN 10
#define LEAF_ALIGN 16
#define ALIGN_UP(ptr, align) ((void *)(((uintptr_t)(ptr) + ((align) - 1)) & ~((align) - 1)))
#define ALIGN_UP_SIZE(size, align) (((size) + ((align) - 1)) & ~((align) - 1))

/**
 * Macros to manipulate pointer tags
 */
#define IS_LEAF(x) (((uintptr_t)x & 1))
#define SET_LEAF(x) ((void *)((uintptr_t)x | 1))
#define LEAF_RAW(x) ((art_leaf *)((void *)((uintptr_t)x & ~1)))

#if defined(__GNUC__) && !defined(__clang__)
#if __STDC_VERSION__ >= 199901L && 402 == (__GNUC__ * 100 + __GNUC_MINOR__)
/*
 * GCC 4.2.2's C99 inline keyword support is pretty broken; avoid. Introduced in
 * GCC 4.2.something, fixed in 4.3.0. So checking for specific major.minor of
 * 4.2 is fine.
 */
#define BROKEN_GCC_C99_INLINE
#endif
#endif

#ifndef CLFLUSH4
#define CLFLUSH4 0
#endif
#ifndef CLFLUSH16
#define CLFLUSH16 0
#endif
#ifndef CLFLUSH48
#define CLFLUSH48 0
#endif
#ifndef CLFLUSH256
#define CLFLUSH256 0
#endif
#define CNT 0             // bookkeeping # count for diff node types
#define HIT_CNT_TOTAL 0   // bookkeeping # hit for diff node types
#define DEPTH_INDI 1      // bookkeeping avg depth for individual node
#define HIT_DIST 1        // for # hit distribution for diff node types
#define LEVEL_ORDER 0     // perform level traversal to collect node type composition
#define STREAM_ACC_ADDR 0 // stream accessed address for each node, should only be enabled for debugging
#define STREAM_ACC_ADDR_256 0
#define ONLINE 1         // online swapping
#define SELF_REF 1       // adding self_ref
#define LEAF_REF 0       // adding self_ref to leaf, no need
#define DFS 0            // do dfs to dump node and path hotness
#define VIS 0            // visualize tree
#define FIRST_TOUCH 0    // measuring first touch
#define STATIC 0         // static placement
#define ENABLE_PROFILE 0 // enable profiling, this cannot be enabled if STATIC is 0

    typedef int (*art_callback)(void *data, const unsigned char *key, uint32_t key_len, void *value);

    typedef struct art_node art_node;
    // typedef struct art_leaf art_leaf;
    /**
     * This struct is included as part
     * of all the various node sizes
     */
    struct art_node
    {
        uint32_t partial_len;                  // 4
        uint8_t type;                          // 1
        uint8_t num_children;                  // 1
        unsigned char partial[MAX_PREFIX_LEN]; // 10
#if HIT_DIST
        int hit_cnt; // 4
#endif
#if DEPTH_INDI
        uint32_t depth; // 4
#endif
#if SELF_REF
        art_node **self_ref; // 8
#endif
#if ONLINE
        int idx_in_arr; // 4
        bool in_local;  // 1 + 3 padding, no needed
#endif
    }; // 32 + 4(online)

    /**
     * Small node with only 4 children
     */
    typedef struct
    {
        art_node n;            // 32
        unsigned char keys[4]; // 4 + 4 padding = 8
        art_node *children[4]; // 4 * 8 = 32, offset: 40
    } art_node4;               // 72

    /**
     * Node with 16 children
     */
    typedef struct
    {
        art_node n;             // 32
        unsigned char keys[16]; // 16 (no padding)
        art_node *children[16]; // 16*8 = 128, offset: 48
    } art_node16;               // 176

    /**
     * Node with 48 children, but
     * a full 256 byte field.
     */
    typedef struct
    {
        art_node n;              // 32
        unsigned char keys[256]; // 256 (no padding)
        art_node *children[48];  // 48*8 = 384, offset: 288
    } art_node48;                // 672

    /**
     * Full node with 256 children
     */
    typedef struct
    {
        art_node n;              // 32
        art_node *children[256]; // 256 * 8 = 2048, offset: 32
    } art_node256;               // 2080

    /**
     * Represents a leaf. These are
     * of arbitrary size, as they include the key.
     */
    typedef struct
    {
        void *value;
        uint32_t key_len;
// #if DEPTH_INDI
//         uint32_t depth;
// #endif
#if LEAF_REF
        art_node **self_ref;
#endif
        unsigned char key[];
    } art_leaf;

    /**
     * Main struct, points to root.
     */
    typedef struct
    {
        art_node *root;
        uint64_t size;
    } art_tree;

    /**
     * Initializes an ART tree
     * @return 0 on success.
     */
    int art_tree_init(art_tree *t, char *wl);

/**
 * DEPRECATED
 * Initializes an ART tree
 * @return 0 on success.
 */
#define init_art_tree(...) art_tree_init(__VA_ARGS__)

    /**
     * Destroys an ART tree
     * @return 0 on success.
     */
    int art_tree_destroy(art_tree *t);

/**
 * DEPRECATED
 * Initializes an ART tree
 * @return 0 on success.
 */
#define destroy_art_tree(...) art_tree_destroy(__VA_ARGS__)

/**
 * Returns the size of the ART tree.
 */
#ifdef BROKEN_GCC_C99_INLINE
#define art_size(t) ((t)->size)
#else
inline uint64_t art_size(art_tree *t)
{
    return t->size;
}
#endif

    /**
     * inserts a new value into the art tree
     * @arg t the tree
     * @arg key the key
     * @arg key_len the length of the key
     * @arg value opaque value.
     * @return null if the item was newly inserted, otherwise
     * the old value pointer is returned.
     */
    void *art_insert(art_tree *t, const unsigned char *key, int key_len, void *value);

    /**
     * inserts a new value into the art tree (not replacing)
     * @arg t the tree
     * @arg key the key
     * @arg key_len the length of the key
     * @arg value opaque value.
     * @return null if the item was newly inserted, otherwise
     * the old value pointer is returned.
     */
    void *art_insert_no_replace(art_tree *t, const unsigned char *key, int key_len, void *value);

    /**
     * Deletes a value from the ART tree
     * @arg t The tree
     * @arg key The key
     * @arg key_len The length of the key
     * @return NULL if the item was not found, otherwise
     * the value pointer is returned.
     */
    void *art_delete(art_tree *t, const unsigned char *key, int key_len);

    /**
     * Searches for a value in the ART tree
     * @arg t The tree
     * @arg key The key
     * @arg key_len The length of the key
     * @return NULL if the item was not found, otherwise
     * the value pointer is returned.
     */
    void *art_search(const art_tree *t, const unsigned char *key, int key_len);

    /**
     * Returns the minimum valued leaf
     * @return The minimum leaf or NULL
     */
    art_leaf *art_minimum(art_tree *t);

    /**
     * Returns the maximum valued leaf
     * @return The maximum leaf or NULL
     */
    art_leaf *art_maximum(art_tree *t);

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
    int art_iter(art_tree *t, art_callback cb, void *data);

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
    int art_iter_prefix(art_tree *t, const unsigned char *prefix, int prefix_len, art_callback cb, void *data);
#if CNT
    extern unsigned long node4_cnt;
    extern unsigned long node16_cnt;
    extern unsigned long node48_cnt;
    extern unsigned long node256_cnt;
    extern unsigned long leaf_cnt;
    void node_cnt_stat();
#endif
#if HIT_CNT_TOTAL
    unsigned long node4_hit_cnt = 0;
    unsigned long node16_hit_cnt = 0;
    unsigned long node48_hit_cnt = 0;
    unsigned long node256_hit_cnt = 0;
    unsigned long leaf_hit_cnt = 0;
    void reset_node_hit_cnt_total();
    void node_hit_cnt_total();
#endif

#if DEPTH_INDI
    extern size_t subtree_inc_func_called;
    extern size_t total_subtree_incremented_nodes;
    typedef struct
    {
        size_t node4_depth_total;
        size_t node4_count;

        size_t node16_depth_total;
        size_t node16_count;

        size_t node48_depth_total;
        size_t node48_count;

        size_t node256_depth_total;
        size_t node256_count;
    } node_depth_stats_t;

    void collect_node_depths(art_node *n, int depth, node_depth_stats_t *stats, FILE *fd);
    void print_avg_node_depths(const node_depth_stats_t *s);
#endif
#if HIT_DIST
    void stream_node_hit_counts_individual(art_node *n, FILE *out);
    void cooling_node_hit_cnt_individual(art_node *n, float factor); // cooling factor: 0 means completely reset the hit_cnt. Less value indicates less weight for history factors
#endif
#if LEVEL_ORDER
    typedef struct node_level_entry
    {
        art_node *node;
        int depth;
        struct node_level_entry *next;
    } node_level_entry;
    void stream_level_distribution(art_node *root, FILE *out);
#endif
    void stream_node_type_addr(art_node *n, FILE *fd);
    // static int check_numa_node(void *addr);

    void init_region(void **base, size_t size, int use_cxl, struct memkind **kind);
    void destroy_region(void *base, size_t size, struct memkind *kind);

    extern void *leaf_base; // mmaped ptr, for mmap and munmap
    extern void *node4_base;
    extern void *node16_base;
    extern void *node48_base;
    extern void *node256_base;
    extern struct memkind *leaf_kind; // memkind ptr
    extern struct memkind *node4_kind;
    extern struct memkind *node16_kind;
    extern struct memkind *node48_kind;
    extern struct memkind *node256_kind;
#if DFS
    void dfs_print_hit_cnt_path(art_node *node, int depth, int *path, void **node_path, FILE *fd);
#endif

#if STATIC || ONLINE
    extern void *node4_local;
    extern void *node4_cxl;
    extern void *node16_local;
    extern void *node16_cxl;
    extern void *node48_local;
    extern void *node48_cxl;
    extern void *node256_local;
    extern void *node256_cxl;
    extern struct memkind *node4_local_kind;
    extern struct memkind *node4_cxl_kind;
    extern struct memkind *node16_local_kind;
    extern struct memkind *node16_cxl_kind;
    extern struct memkind *node48_local_kind;
    extern struct memkind *node48_cxl_kind;
    extern struct memkind *node256_local_kind;
    extern struct memkind *node256_cxl_kind;
    extern struct memkind *local_kinds[4];
    extern struct memkind *cxl_kinds[4];
#endif
#if ENABLE_PROFILE
    extern int node4_local_cnt;
    extern int node4_cxl_cnt;
    extern int node16_local_cnt;
    extern int node16_cxl_cnt;
    extern int node48_local_cnt;
    extern int node48_cxl_cnt;
    extern int node256_local_cnt;
    extern int node256_cxl_cnt;
    extern int node4_mis_placed;
    extern int node16_mis_placed;
    extern int node48_mis_placed;
    extern int node256_mis_placed;
    extern int *local_cnt[4];
    extern int *cxl_cnt[4];
    extern int *misplaced_cnt[4];
#endif
    extern FILE *log_fd;

#if STREAM_ACC_ADDR || STREAM_ACC_ADDR_256
    FILE *acc_fd;
    int start_acc_streaming = 0;
#endif
#if ONLINE
    extern bool node4_local_full;
    extern bool node16_local_full;
    extern bool node48_local_full;
    extern bool node256_local_full;
    extern art_node **node4_hot;
    extern art_node **node4_cold;
    extern art_node **node16_hot;
    extern art_node **node16_cold;
    extern art_node **node48_hot;
    extern art_node **node48_cold;
    extern art_node **node256_hot;
    extern art_node **node256_cold;
    extern int node4_local_alloc_cnt;
    extern int node4_cxl_alloc_cnt;
    extern int node16_local_alloc_cnt;
    extern int node16_cxl_alloc_cnt;
    extern int node48_local_alloc_cnt;
    extern int node48_cxl_alloc_cnt;
    extern int node256_local_alloc_cnt;
    extern int node256_cxl_alloc_cnt;
    extern int traverse_cnt;
    art_node *tiered_calloc(bool *local_full,
                            memkind_t local_kind, memkind_t cxl_kind,
                            size_t size,
                            art_node **node_hot_arr, int *hot_count,
                            art_node **node_cold_arr, int *cold_count);
    // art_node *tiered_calloc(bool *local_full, struct memkind *local_kind, struct memkind *cxl_kind, size_t size, uint8_t type); // for lazy tree traversal
    static void sort_hotness(art_node **alloced_nodes, int alloced_cnt, bool descending);
    static void del_node_from_arr(art_node *n, int *local_alloc_cnt, int *cxl_alloc_cnt, art_node **hot_arr, art_node **cold_arr); // remove node from node array when freed
    static void swap_hot_cold_nodes(art_node **hot_node_arr, int hot_node_count, art_node **cold_node_arr, int cold_node_count);
    static void swap_hot_cold_nodes_bulk(art_node **hot_node_arr, int hot_node_count, art_node **cold_node_arr, int cold_node_count);
    // static void del_node_from_arr(art_node *n, struct memkind *kind);
    void sort_all_hotness();
    void traverse_tree_populate_min_heap(art_node *n);
    void populate_min_heap(art_node *n);
    void print_min_heap_stat();
    void reset_min_heap();
    void ins_node_to_set(art_node *n, uint8_t type);
    void del_node_from_set(art_node *n, struct memkind *kind);
    void get_top_k_and_swap();
    void get_top_k_by_hit_cnt(art_node **out_arr, int *out_k, art_node **arr, int N, int k, bool ascending);

    // for LRU
    // extern art_node *LRU_node4[HOT_CACHE_LIMIT];
    void hot_cache_record_access(art_node *n);
    int hot_cache_snapshot(art_node **out_arr, int max_size);
    void hot_cache_reset();

    void reset_ring_buffer();
    void sample_to_ring_buffer(art_node *n);
    void get_ring_buffer_snapshot(art_node **out_arr, int *out_cnt);
#endif
#if SELF_REF
    void dump_self_ref_json(FILE *out, art_node *n, void *parent_child_ptr);
    // void dump_self_ref_json(FILE *out, art_node *n);
#endif
#if STATIC
    static void load_static_metrics(char *wl);
    static void free_static_metrics();
    extern int static_metrics_line_cnt;
    extern int **matrix;
#endif
    int show_stat();

#ifdef __cplusplus
}
#endif

#endif
