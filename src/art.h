#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#ifndef ART_H
#define ART_H

#ifdef __cplusplus
extern "C" {
#endif

#include "node_allocator.h"
#define NODE4 1
#define NODE16 2
#define NODE48 3
#define NODE256 4

#define MAX_PREFIX_LEN 10

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

#define CUS_ALLOC 1         // custom allocator for inner nodes
#define LEAF_CUS_ALLOC 1    // custom allocator for leaf nodes
#define LEAF_DISTRIBUTION 0 // show leaf_lens distribution
#define CNT 0               // bookkeep node count
#define HIT_CNT_TOTAL 0     // bookkeep total hit count
#define BOOKKEEP 1          // add metadata for inner nodes
#define HIT_CNT 0           // bookkeep hit_cnt for each inner node
#define DEPTH 0             // bookkeep depth
#define SELF_REF 0          // bookkeep pointer to parent's children slot
#define LEAF_CENTRIC 1      // leaf-centric sampling, extra field
#define THREAD 1            // concurrency control with versioning

#if BOOKKEEP
#define PTR_MASK ((1ULL << 48) - 1)
#define DEPTH_SHIFT 48
#define DEPTH_MASK ((1ULL << 11) - 1)
#define IS_LOCAL_SHIFT 59
#define IS_LOCAL_MASK 1ULL
#endif
#if CNT
extern unsigned long node4_cnt;
extern unsigned long node16_cnt;
extern unsigned long node48_cnt;
extern unsigned long node256_cnt;
extern unsigned long leaf_cnt;
void node_cnt_stat();
#endif

#if HIT_CNT_TOTAL
extern unsigned long node4_hit_cnt;
extern unsigned long node16_hit_cnt;
extern unsigned long node48_hit_cnt;
extern unsigned long node256_hit_cnt;
extern unsigned long leaf_hit_cnt;
void reset_node_hit_cnt_total();
void node_hit_cnt_total();
#endif
typedef int (*art_callback)(void *data, const unsigned char *key,
                            uint32_t key_len, void *value);

/**
 * This struct is included as part
 * of all the various node sizes
 */
typedef struct {
  uint32_t partial_len;                  // 4
  uint8_t type;                          // 1
  uint8_t num_children;                  // 1
  unsigned char partial[MAX_PREFIX_LEN]; // 10
#if BOOKKEEP
  // |63         60|59       |58         48|47          0|
  // |  unused (4) |is_local |  depth(11)  |  ptr (48b)  |
  uint64_t loc_depth_ptr; // 8
#endif
#if HIT_CNT
  uint16_t hit_cnt; // 2
#endif
#if THREAD
  uint32_t
      version; // 4 bytes - version for optimistic reads + migration status bit
#endif
} art_node;

#if THREAD
// Version field layout: |31 bits version|1 bit migration status|
#define VERSION_MASK ((1U << 31) - 1)   // 31 bits for version
#define MIGRATION_STATUS_BIT (1U << 31) // 1 bit for migration status

// Version manipulation macros
static inline uint32_t get_node_version(art_node *n) {
  return n->version & VERSION_MASK;
}

static inline void set_node_version(art_node *n, uint32_t version) {
  n->version = (n->version & MIGRATION_STATUS_BIT) | (version & VERSION_MASK);
}

static inline void increment_node_version(art_node *n) {
  uint32_t current_version = get_node_version(n);
  set_node_version(n, current_version + 1);
}

static inline bool get_migration_status(art_node *n) {
  return (n->version & MIGRATION_STATUS_BIT) != 0;
}

static inline void set_migration_status(art_node *n, bool migrating) {
  if (migrating) {
    n->version |= MIGRATION_STATUS_BIT;
  } else {
    n->version &= ~MIGRATION_STATUS_BIT;
  }
}
// Fast version access macros
#define FAST_GET_VERSION(n) ((n)->version & VERSION_MASK)
#define FAST_GET_MIGRATION(n) (((n)->version & MIGRATION_STATUS_BIT) != 0)
#define FAST_INCREMENT_VERSION(n)                                              \
  ((n)->version = ((n)->version & MIGRATION_STATUS_BIT) |                      \
                  (((n)->version & VERSION_MASK) + 1))
#endif // THREAD

/**
 * Small node with only 4 children
 */
typedef struct {
  art_node n;
  unsigned char keys[4];
  art_node *children[4];
} art_node4;

/**
 * Node with 16 children
 */
typedef struct {
  art_node n;
  unsigned char keys[16];
  art_node *children[16];
} art_node16;

/**
 * Node with 48 children, but
 * a full 256 byte field.
 */
typedef struct {
  art_node n;
  unsigned char keys[256];
  art_node *children[48];
} art_node48;

/**
 * Full node with 256 children
 */
typedef struct {
  art_node n;
  art_node *children[256];
} art_node256;

/**
 * Represents a leaf. These are
 * of arbitrary size, as they include the key.
 */
typedef struct {
  void *value;
  uint32_t key_len;
#if LEAF_CENTRIC
  uint64_t acc_parent_compact;
#endif
  unsigned char key[];
} art_leaf;
/**
 * Main struct, points to root.
 */
typedef struct {
  art_node *root;
  uint64_t size;
#if THREAD
  pthread_rwlock_t tree_lock;
#endif
} art_tree;
#if THREAD
typedef struct {
  pthread_rwlock_t node_lock;   // Lock for the node itself
  pthread_rwlock_t parent_lock; // Lock for the parent node
  art_node *parent;             // Pointer to parent node
} art_node_lock_t;

// Lock management structure
typedef struct lock_entry {
  art_node *node;
  art_node_lock_t lock_info;
  struct lock_entry *next;
} lock_entry_t;

// Lock table structure
typedef struct {
  lock_entry_t **buckets;
  size_t size;
  pthread_mutex_t table_lock;
} lock_table_t;

// Function declarations for lock management
lock_table_t *create_lock_table(size_t size);
void destroy_lock_table(lock_table_t *table);
art_node_lock_t *get_node_lock_info(art_node *node, lock_table_t *table);
void cleanup_node_locks(art_node *node, lock_table_t *table);

// Background worker management
typedef struct {
  pthread_t worker_thread;
  pthread_mutex_t worker_mutex;
  pthread_cond_t worker_cond;
  bool should_stop;
  art_tree *tree;
  lock_table_t *lock_table;
} background_worker_t;

// Background worker functions
background_worker_t *start_background_worker(art_tree *tree,
                                             lock_table_t *lock_table);
void stop_background_worker(background_worker_t *worker);
void *background_worker_thread(void *arg);

// Sampling function declaration
void sampling(art_tree *tree, lock_table_t *lock_table);

// Migration function declaration
int migrate_node(art_tree *t, art_node *node, art_node *parent);

// Thread-safe versions of operations
void *art_insert_thread_safe(art_tree *t, const unsigned char *key, int key_len,
                             void *value);
void *art_delete_thread_safe(art_tree *t, const unsigned char *key,
                             int key_len);
void *art_search_thread_safe(const art_tree *t, const unsigned char *key,
                             int key_len);
void *art_search_optimistic(const art_tree *t, const unsigned char *key,
                            int key_len);
void *art_search_internal(const art_tree *t, const unsigned char *key,
                          int key_len);
static void *art_search_pessimistic(const art_tree *t, const unsigned char *key,
                                    int key_len);

// Global variables (extern declarations)
extern lock_table_t *global_lock_table;
extern background_worker_t *global_worker;
#endif // THREAD

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
inline uint64_t art_size(art_tree *t) { return t->size; }
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
void *art_insert(art_tree *t, const unsigned char *key, int key_len,
                 void *value);

/**
 * inserts a new value into the art tree (not replacing)
 * @arg t the tree
 * @arg key the key
 * @arg key_len the length of the key
 * @arg value opaque value.
 * @return null if the item was newly inserted, otherwise
 * the old value pointer is returned.
 */
void *art_insert_no_replace(art_tree *t, const unsigned char *key, int key_len,
                            void *value);

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
int art_iter_prefix(art_tree *t, const unsigned char *prefix, int prefix_len,
                    art_callback cb, void *data);

// additional symbols
#if CUS_ALLOC
node_allocator na_node4;
node_allocator na_node16;
node_allocator na_node48;
node_allocator na_node256;
#endif
#if LEAF_CUS_ALLOC
#define LEAF_SIZE_CLASSES 9
node_allocator na_leaf_16;    // 9-16 bytes
node_allocator na_leaf_24;    // 17-24 bytes
node_allocator na_leaf_32;    // 25-32 bytes
node_allocator na_leaf_40;    // 33-40 bytes
node_allocator na_leaf_48;    // 41-48 bytes
node_allocator na_leaf_54;    // 49-54 bytes
node_allocator na_leaf_60;    // 55-60 bytes
node_allocator na_leaf_large; // 61-72 bytes
static int get_leaf_size_class(size_t total_size);
static node_allocator *get_leaf_allocator(int size_class);
static node_allocator *find_leaf_allocator(art_leaf *leaf);
#endif
// Set pointer (preserving depth and is_local)
static inline void set_ptr(art_node *n, void *ptr) {
  uint64_t ptr_val = (uint64_t)ptr & PTR_MASK;
  n->loc_depth_ptr = (n->loc_depth_ptr & ~PTR_MASK) | ptr_val;
  // n->self_ref = ptr;
}

static inline void *get_ptr(art_node *n) {
  return (void *)(n->loc_depth_ptr & PTR_MASK);
  // return n->self_ref;
}
#if BOOKKEEP

static inline void set_depth(art_node *n, uint16_t depth) {
  n->loc_depth_ptr &= ~(DEPTH_MASK << DEPTH_SHIFT);
  n->loc_depth_ptr |= ((uint64_t)(depth & DEPTH_MASK)) << DEPTH_SHIFT;
}

static inline void decrement_depth(art_node *n) {
  // uint64_t val = n->loc_depth_ptr;
  // uint16_t depth = (val >> DEPTH_SHIFT) & DEPTH_MASK;

  // // Prevent underflow
  // if (depth > 0)
  // {
  //     depth -= 1;
  // }

  // // Clear existing depth bits and update with decremented value
  // val &= ~(DEPTH_MASK << DEPTH_SHIFT);
  // val |= ((uint64_t)depth) << DEPTH_SHIFT;

  // n->loc_depth_ptr = val;
  uint64_t d = (n->loc_depth_ptr >> DEPTH_SHIFT) & DEPTH_MASK;
  d = (d - 1) & DEPTH_MASK; // wraps around to 2047 if d==0, avoids branching
  n->loc_depth_ptr =
      (n->loc_depth_ptr & ~(DEPTH_MASK << DEPTH_SHIFT)) | (d << DEPTH_SHIFT);
}

static inline void increment_depth(art_node *n) {
  uint64_t d = (n->loc_depth_ptr >> DEPTH_SHIFT) & DEPTH_MASK;
  d = (d + 1) & DEPTH_MASK; // wraps around at 2048, respects 11-bit limit
  n->loc_depth_ptr =
      (n->loc_depth_ptr & ~(DEPTH_MASK << DEPTH_SHIFT)) | (d << DEPTH_SHIFT);
}

static inline uint16_t get_depth(art_node *n) {
  return (uint16_t)((n->loc_depth_ptr >> DEPTH_SHIFT) & DEPTH_MASK);
}

static inline void set_is_local(art_node *n, bool is_local) {
  n->loc_depth_ptr &= ~(IS_LOCAL_MASK << IS_LOCAL_SHIFT);
  n->loc_depth_ptr |= ((uint64_t)is_local & IS_LOCAL_MASK) << IS_LOCAL_SHIFT;
}

static inline bool get_is_local(art_node *n) {
  return (bool)((n->loc_depth_ptr >> IS_LOCAL_SHIFT) & IS_LOCAL_MASK);
}
#endif // BOOKKEEP

#if DEPTH
static void increment_subtree_depth(art_node *n);
void collect_node_depths(art_node *n, int depth, FILE *fd);
#endif

#if SELF_REF
static inline void refresh_self_refs(art_node *n, int start, int end);
static void fix_children_self_ref(void **children, int count);
void dump_self_ref_json(FILE *out, art_node *n, void *parent_child_ptr);
#endif

#if LEAF_CENTRIC
// Getter and setter for acc_parent_compact field
static inline uint64_t get_leaf_access_count(art_leaf *leaf) {
  return (leaf->acc_parent_compact >> 48) & 0xFFFF;
}

static inline void set_leaf_access_count(art_leaf *leaf, uint64_t count) {
  leaf->acc_parent_compact =
      (leaf->acc_parent_compact & 0xFFFFFFFFFFFF) | ((count & 0xFFFF) << 48);
}

static inline void increment_leaf_access_count(art_leaf *leaf) {
  uint64_t current_count = get_leaf_access_count(leaf);
  set_leaf_access_count(leaf, current_count + 1);
}

static inline art_node *get_leaf_parent_ptr(art_leaf *leaf) {
  return (art_node *)(leaf->acc_parent_compact & 0xFFFFFFFFFFFF);
}

static inline void set_leaf_parent_ptr(art_leaf *leaf, art_node *parent) {
  leaf->acc_parent_compact = (leaf->acc_parent_compact & 0xFFFF000000000000) |
                             ((uint64_t)parent & 0xFFFFFFFFFFFF);
}
#endif // LEAF_CENTRIC

#ifdef __cplusplus
}
#endif

#endif