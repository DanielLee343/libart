#include <stdint.h>
#include <stddef.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <memkind.h>
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

    void *slab_base;
    void *bump_ptr;
    void *node256_base;
    void *node256_ptr;
    void *node48_base;
    void *node48_ptr;
    void *node16_base;
    void *node16_ptr;
    void *node4_base;
    void *node4_ptr;

#define MAX_PREFIX_LEN 10
#define LEAF_ALIGN 16
#define ALIGN_UP(ptr, align) ((void *)(((uintptr_t)(ptr) + ((align) - 1)) & ~((align) - 1)))
#define ALIGN_UP_SIZE(size, align) (((size) + ((align) - 1)) & ~((align) - 1))

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
#define CNT 0             // bookkeeping # count for diff node types
#define HIT_CNT_TOTAL 0   // bookkeeping # hit for diff node types
#define DEPTH 0           // bookkeeping avg depth for diff node types
#define HIT_DIST 0        // for # hit distribution for diff node types
#define LEVEL_ORDER 0     // perform level traversal to collect node type composition
#define STATIC_DIST 0     // perform static placement based on depth
#define STREAM_ACC_ADDR 0 // stream accessed address for each node, should only be enabled for debugging
#define ONLINE 1          // online swapping

    typedef int (*art_callback)(void *data, const unsigned char *key, uint32_t key_len, void *value);

    /**
     * This struct is included as part
     * of all the various node sizes
     */
    typedef struct
    {
        uint32_t partial_len;
        uint8_t type;
        uint8_t num_children;
        unsigned char partial[MAX_PREFIX_LEN];
#if HIT_DIST
        int hit_cnt;
#endif
#if DEPTH
        int depth;
#endif
        // #if STATIC_DIST
        //         struct memkind *alloc_kind;
        // #endif
    } art_node;

    /**
     * Small node with only 4 children
     */
    typedef struct
    {
        art_node n;
        unsigned char keys[4];
        art_node *children[4];
    } art_node4;

    /**
     * Node with 16 children
     */
    typedef struct
    {
        art_node n;
        unsigned char keys[16];
        art_node *children[16];
    } art_node16;

    /**
     * Node with 48 children, but
     * a full 256 byte field.
     */
    typedef struct
    {
        art_node n;
        unsigned char keys[256];
        art_node *children[48];
    } art_node48;

    /**
     * Full node with 256 children
     */
    typedef struct
    {
        art_node n;
        art_node *children[256];
    } art_node256;

    /**
     * Represents a leaf. These are
     * of arbitrary size, as they include the key.
     */
    typedef struct
    {
        void *value;
        uint32_t key_len;
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
    int art_tree_init(art_tree *t);

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
    unsigned long node4_cnt = 0;
    unsigned long node16_cnt = 0;
    unsigned long node48_cnt = 0;
    unsigned long node256_cnt = 0;
    unsigned long leaf_cnt = 0;
    void node_cnt_stat();
#endif
#if HIT_CNT_TOTAL
    unsigned long node4_hit_cnt = 0;
    unsigned long node16_hit_cnt = 0;
    unsigned long node48_hit_cnt = 0;
    unsigned long node256_hit_cnt = 0;
    unsigned long leaf_hit_cnt = 0;
    void node_hit_cnt_total();
    void reset_node_hit_cnt_total();
#endif

#if DEPTH
#define NODE_DEPTH(n) (((art_node *)(n))->depth)
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
    static void swap_art_nodes(art_node **n0, art_node **n1); // not used
    // static int check_numa_node(void *addr);
    size_t total_leaf_count = 0;

    void init_region(void **base, size_t size, int use_cxl, struct memkind **kind);
    void destroy_region(void *base, size_t size, struct memkind *kind);

    void *leaf_base = NULL; // mmaped ptr, for mmap and munmap
    void *node4_base = NULL;
    void *node16_base = NULL;
    void *node48_base = NULL;
    void *node256_base = NULL;
    struct memkind *leaf_kind = NULL; // memkind ptr
    struct memkind *node4_kind = NULL;
    struct memkind *node16_kind = NULL;
    struct memkind *node48_kind = NULL;
    struct memkind *node256_kind = NULL;
#if STATIC_DIST || ONLINE
    void distribute_nodes(art_node *n, art_node **ref, int curr_depth);
    void print_node_move_stat();
    // leaf
    void *leaf_local = NULL;                // for local mmaped ptr
    void *leaf_cxl = NULL;                  // for cxl mmaped ptr
    struct memkind *leaf_local_kind = NULL; // for local memkind ptr
    struct memkind *leaf_cxl_kind = NULL;   // for cxl memkind ptr
    // node4
    void *node4_local = NULL;
    void *node4_cxl = NULL;
    struct memkind *node4_local_kind = NULL;
    struct memkind *node4_cxl_kind = NULL;
    // node16
    void *node16_local = NULL;
    void *node16_cxl = NULL;
    struct memkind *node16_local_kind = NULL;
    struct memkind *node16_cxl_kind = NULL;
    // node48
    void *node48_local = NULL;
    void *node48_cxl = NULL;
    struct memkind *node48_local_kind = NULL;
    struct memkind *node48_cxl_kind = NULL;
    // node256
    void *node256_local = NULL;
    void *node256_cxl = NULL;
    struct memkind *node256_local_kind = NULL;
    struct memkind *node256_cxl_kind = NULL;
    // counting move
    int leaf_moved_local = 0;
    int leaf_moved_cxl = 0;
    int node4_moved_local = 0;
    int node4_moved_cxl = 0;
    int node16_moved_local = 0;
    int node16_moved_cxl = 0;
    int node48_moved_local = 0;
    int node48_moved_cxl = 0;
    int node256_moved_local = 0;
    int node256_moved_cxl = 0;
#endif
#if STREAM_ACC_ADDR
    FILE *acc_fd;
    int start_acc_streaming = 0;
#endif
#ifdef __cplusplus
}
#endif

#endif
