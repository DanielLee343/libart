#include "art.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#ifdef __i386__
#include <emmintrin.h>
#else
#ifdef __amd64__
#include <emmintrin.h>
#endif
#endif

/**
 * Macros to manipulate pointer tags
 */
#define IS_LEAF(x) (((uintptr_t)x & 1))
#define SET_LEAF(x) ((void *)((uintptr_t)x | 1))
#define LEAF_RAW(x) ((art_leaf *)((void *)((uintptr_t)x & ~1)))

#define PAGE_SIZE 4096UL
#define NUM_4_PAGE 496396UL
#define NUM_16_PAGE 240034UL
#define NUM_48_PAGE 247328UL
#define NUM_256_PAGE 1300000UL
#if LEAF_CUS_ALLOC
// email_workloadbigger_tail
// #define NUM_LEAF_16 100
// #define NUM_LEAF_24 600
// #define NUM_LEAF_32 2181000
// #define NUM_LEAF_40 8745000
// #define NUM_LEAF_48 1069000
// #define NUM_LEAF_54 75000
// #define NUM_LEAF_60 10
// #define NUM_LEAF_LARGE 10

// email_workloadbigger_tail w/ extra field
#define NUM_LEAF_16 10
#define NUM_LEAF_24 10
#define NUM_LEAF_32 600
#define NUM_LEAF_40 2181000
#define NUM_LEAF_48 8745000
#define NUM_LEAF_54 1004000
#define NUM_LEAF_60 140000
#define NUM_LEAF_LARGE 10

// email_workloadc_ext_10
// #define NUM_LEAF_16 10
// #define NUM_LEAF_24 3000
// #define NUM_LEAF_32 38000
// #define NUM_LEAF_40 20849000
// #define NUM_LEAF_48 66232357
// #define NUM_LEAF_54 26300000
// #define NUM_LEAF_60 5517000
// #define NUM_LEAF_LARGE 639000

// email_workloadc_ext_10 w/ extra field
// #define NUM_LEAF_16 10
// #define NUM_LEAF_24 10
// #define NUM_LEAF_32 3000
// #define NUM_LEAF_40 38000
// #define NUM_LEAF_48 20849000
// #define NUM_LEAF_54 49924000
// #define NUM_LEAF_60 37125000
// #define NUM_LEAF_LARGE 11631000
#endif // LEAF_CUS_ALLOC

#if LEAF_DISTRIBUTION
int num_leaf_16;
int num_leaf_24;
int num_leaf_32;
int num_leaf_40;
int num_leaf_48;
int num_leaf_54;
int num_leaf_60;
int num_leaf_large;
int max_leaf_size;
#endif

/**
 * Allocates a node of the given type,
 * initializes to zero and sets the type.
 */
static art_node *alloc_node(uint8_t type) {
  art_node *n;
  switch (type) {
  case NODE4:
#if CUS_ALLOC
    n = (art_node *)alloc_node_cus(&na_node4);
#else
    n = (art_node *)calloc(1, sizeof(art_node4));
#endif
#if CNT
    node4_cnt++;
#endif
    break;
  case NODE16:
#if CUS_ALLOC
    n = (art_node *)alloc_node_cus(&na_node16);
#else
    n = (art_node *)calloc(1, sizeof(art_node16));
#endif
#if CNT
    node16_cnt++;
#endif
    break;
  case NODE48:
#if CUS_ALLOC
    n = (art_node *)alloc_node_cus(&na_node48);
#else
    n = (art_node *)calloc(1, sizeof(art_node48));
#endif
#if CNT
    node48_cnt++;
#endif
    break;
  case NODE256:
#if CUS_ALLOC
    n = (art_node *)alloc_node_cus(&na_node256);
#else
    n = (art_node *)calloc(1, sizeof(art_node256));
#endif
#if CNT
    node256_cnt++;
#endif
    break;
  default:
    abort();
  }
  n->type = type;
#if THREAD
  set_node_version(n, 0);         // Initialize version to 0
  set_migration_status(n, false); // Initialize migration status to false
#endif
  return n;
}

/**
 * Initializes an ART tree
 * @return 0 on success.
 */
int art_tree_init(art_tree *t) {
  t->root = NULL;
  t->size = 0;
#if CUS_ALLOC
  init_allocator(&na_node4,
                 (size_t)(NUM_4_PAGE * PAGE_SIZE) / sizeof(art_node4),
                 sizeof(art_node4), 0);
  init_allocator(&na_node16,
                 (size_t)(NUM_16_PAGE * PAGE_SIZE) / sizeof(art_node16),
                 sizeof(art_node16), 0);
  init_allocator(&na_node48,
                 (size_t)(NUM_48_PAGE * PAGE_SIZE) / sizeof(art_node48),
                 sizeof(art_node48), 0);
  init_allocator(&na_node256,
                 (size_t)(NUM_256_PAGE * PAGE_SIZE) / sizeof(art_node256),
                 sizeof(art_node256), 0);
  register_allocator(&na_node4);
  register_allocator(&na_node16);
  register_allocator(&na_node48);
  register_allocator(&na_node256);
#endif
#if LEAF_CUS_ALLOC
  init_allocator(&na_leaf_16, NUM_LEAF_16, 16, 0);
  init_allocator(&na_leaf_24, NUM_LEAF_24, 24, 0);
  init_allocator(&na_leaf_32, NUM_LEAF_32, 32, 0);
  init_allocator(&na_leaf_40, NUM_LEAF_40, 40, 0);
  init_allocator(&na_leaf_48, NUM_LEAF_48, 48, 0);
  init_allocator(&na_leaf_54, NUM_LEAF_54, 54, 0);
  init_allocator(&na_leaf_60, NUM_LEAF_60, 60, 0);
  init_allocator(&na_leaf_large, NUM_LEAF_LARGE, 72, 0);
  register_allocator(&na_leaf_16);
  register_allocator(&na_leaf_24);
  register_allocator(&na_leaf_32);
  register_allocator(&na_leaf_40);
  register_allocator(&na_leaf_48);
  register_allocator(&na_leaf_54);
  register_allocator(&na_leaf_60);
  register_allocator(&na_leaf_large);
#endif
#if THREAD
  pthread_rwlock_init(&t->tree_lock, NULL);

  // Create global lock table
  global_lock_table = create_lock_table(1024); // Adjust size as needed
  if (!global_lock_table) {
    pthread_rwlock_destroy(&t->tree_lock);
    return -1;
  }

  // Start background worker
  global_worker = start_background_worker(t, global_lock_table);
  if (!global_worker) {
    destroy_lock_table(global_lock_table);
    pthread_rwlock_destroy(&t->tree_lock);
    return -1;
  }
#endif

  return 0;
}

// Recursively destroys the tree
static void destroy_node(art_node *n) {
  // Break if null
  if (!n)
    return;

  // Special case leafs
  if (IS_LEAF(n)) {
    art_leaf *leaf = LEAF_RAW(n);
#if LEAF_CUS_ALLOC
    node_allocator *na = find_leaf_allocator(leaf);
    if (na) {
      free_node(na, leaf);
    } else {
      free(leaf);
    }
#else
    free(leaf);
#endif
#if CNT
    leaf_cnt--;
#endif
    return;
  }

  // Handle each node type
  int i, idx;
  union {
    art_node4 *p1;
    art_node16 *p2;
    art_node48 *p3;
    art_node256 *p4;
  } p;
  switch (n->type) {
  case NODE4:
    p.p1 = (art_node4 *)n;
    for (i = 0; i < n->num_children; i++) {
      destroy_node(p.p1->children[i]);
    }
#if CNT
    node4_cnt--;
#endif
    break;

  case NODE16:
    p.p2 = (art_node16 *)n;
    for (i = 0; i < n->num_children; i++) {
      destroy_node(p.p2->children[i]);
    }
#if CNT
    node16_cnt--;
#endif
    break;

  case NODE48:
    p.p3 = (art_node48 *)n;
    for (i = 0; i < 256; i++) {
      idx = ((art_node48 *)n)->keys[i];
      if (!idx)
        continue;
      destroy_node(p.p3->children[idx - 1]);
    }
#if CNT
    node48_cnt--;
#endif
    break;

  case NODE256:
    p.p4 = (art_node256 *)n;
    for (i = 0; i < 256; i++) {
      if (p.p4->children[i])
        destroy_node(p.p4->children[i]);
    }
#if CNT
    node256_cnt--;
#endif
    break;

  default:
    abort();
  }

// Free ourself on the way up
#if CUS_ALLOC
  free_node_auto((void *)n);
#else
  free(n);
#endif
}

/**
 * Destroys an ART tree
 * @return 0 on success.
 */
int art_tree_destroy(art_tree *t) {
  destroy_node(t->root);
#if CUS_ALLOC
  destroy_allocator(&na_node4);
  destroy_allocator(&na_node16);
  destroy_allocator(&na_node48);
  destroy_allocator(&na_node256);
#endif
#if LEAF_CUS_ALLOC
  destroy_allocator(&na_leaf_16);
  destroy_allocator(&na_leaf_24);
  destroy_allocator(&na_leaf_32);
  destroy_allocator(&na_leaf_40);
  destroy_allocator(&na_leaf_48);
  destroy_allocator(&na_leaf_54);
  destroy_allocator(&na_leaf_60);
  destroy_allocator(&na_leaf_large);
#endif
#if LEAF_DISTRIBUTION
  printf("num_leaf_16: %d\n", num_leaf_16);
  printf("num_leaf_24: %d\n", num_leaf_24);
  printf("num_leaf_32: %d\n", num_leaf_32);
  printf("num_leaf_40: %d\n", num_leaf_40);
  printf("num_leaf_48: %d\n", num_leaf_48);
  printf("num_leaf_54: %d\n", num_leaf_54);
  printf("num_leaf_60: %d\n", num_leaf_60);
  printf("num_leaf_large: %d\n", num_leaf_large);
  printf("max_leaf_size: %d\n", max_leaf_size);
#endif
#if THREAD
  // Stop background worker first
  if (global_worker) {
    stop_background_worker(global_worker);
    global_worker = NULL;
  }

  // Clean up lock table
  if (global_lock_table) {
    destroy_lock_table(global_lock_table);
    global_lock_table = NULL;
  }

  pthread_rwlock_destroy(&t->tree_lock);
#endif
  return 0;
}

/**
 * Returns the size of the ART tree.
 */

#ifndef BROKEN_GCC_C99_INLINE
extern inline uint64_t art_size(art_tree *t);
#endif

static art_node **find_child(art_node *n, unsigned char c) {
  int i, mask, bitfield;
  union {
    art_node4 *p1;
    art_node16 *p2;
    art_node48 *p3;
    art_node256 *p4;
  } p;
  switch (n->type) {
  case NODE4:
    p.p1 = (art_node4 *)n;
    for (i = 0; i < n->num_children; i++) {
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
      for (i = 0; i < 16; ++i) {
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
static inline int min(int a, int b) { return (a < b) ? a : b; }

/**
 * Returns the number of prefix characters shared between
 * the key and node.
 */
static int check_prefix(const art_node *n, const unsigned char *key,
                        int key_len, int depth) {
  int max_cmp = min(min(n->partial_len, MAX_PREFIX_LEN), key_len - depth);
  int idx;
  for (idx = 0; idx < max_cmp; idx++) {
    if (n->partial[idx] != key[depth + idx])
      return idx;
  }
  return idx;
}

/**
 * Checks if a leaf matches
 * @return 0 on success.
 */
static int leaf_matches(const art_leaf *n, const unsigned char *key,
                        int key_len, int depth) {
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
void *art_search(const art_tree *t, const unsigned char *key, int key_len) {
#if THREAD
  return art_search_optimistic(t, key, key_len);
#else
  art_node **child;
  art_node *n = t->root;
  int prefix_len, depth = 0;
  while (n) {
    // Might be a leaf
    if (IS_LEAF(n)) {
#if HIT_CNT_TOTAL
      leaf_hit_cnt++;
#endif
      n = (art_node *)LEAF_RAW(n);
      // Check if the expanded path matches
      if (!leaf_matches((art_leaf *)n, key, key_len, depth)) {
#if LEAF_CENTRIC
        increment_leaf_access_count((art_leaf *)n);
#endif
        return ((art_leaf *)n)->value;
      }
      return NULL;
    }
#if HIT_CNT
    n->hit_cnt++;
#endif
#if HIT_CNT_TOTAL
    switch (n->type) {
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

    // Bail if the prefix does not match
    if (n->partial_len) {
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
#endif // THREAD
}

// Find the minimum leaf under a node
static art_leaf *minimum(const art_node *n) {
  // Handle base cases
  if (!n)
    return NULL;
  if (IS_LEAF(n))
    return LEAF_RAW(n);

  int idx;
  switch (n->type) {
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
static art_leaf *maximum(const art_node *n) {
  // Handle base cases
  if (!n)
    return NULL;
  if (IS_LEAF(n))
    return LEAF_RAW(n);

  int idx;
  switch (n->type) {
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
art_leaf *art_minimum(art_tree *t) { return minimum((art_node *)t->root); }

/**
 * Returns the maximum valued leaf
 */
art_leaf *art_maximum(art_tree *t) { return maximum((art_node *)t->root); }

static art_leaf *make_leaf(const unsigned char *key, int key_len, void *value) {
  size_t total_size = sizeof(art_leaf) + key_len;
#if LEAF_DISTRIBUTION
  if (total_size <= 16)
    num_leaf_16++;
  else if (total_size <= 24)
    num_leaf_24++;
  else if (total_size <= 32)
    num_leaf_32++;
  else if (total_size <= 40)
    num_leaf_40++;
  else if (total_size <= 48)
    num_leaf_48++;
  else if (total_size <= 54)
    num_leaf_54++;
  else if (total_size <= 60)
    num_leaf_60++;
  else {
    num_leaf_large++;
    if (total_size > max_leaf_size)
      max_leaf_size = total_size;
  }
#endif // LEAF_DISTRIBUTION
#if LEAF_CUS_ALLOC
  int size_class = get_leaf_size_class(total_size);
  //   printf("leaf_size: %zu\n", total_size);
  node_allocator *na = get_leaf_allocator(size_class);
  art_leaf *l = (art_leaf *)alloc_node_cus(na);
#else
  art_leaf *l = (art_leaf *)calloc(1, sizeof(art_leaf) + key_len);
#endif
  l->value = value;
  l->key_len = key_len;
  memcpy(l->key, key, key_len);
#if LEAF_CENTRIC
  // Initialize access count to 0 and parent pointer to NULL
  l->acc_parent_compact = 0;
#endif
#if CNT
  leaf_cnt++;
#endif
  return l;
}

static int longest_common_prefix(art_leaf *l1, art_leaf *l2, int depth) {
  int max_cmp = min(l1->key_len, l2->key_len) - depth;
  int idx;
  for (idx = 0; idx < max_cmp; idx++) {
    if (l1->key[depth + idx] != l2->key[depth + idx])
      return idx;
  }
  return idx;
}

static void copy_header(art_node *dest, art_node *src, void *ref) {
  dest->num_children = src->num_children;
  dest->partial_len = src->partial_len;
#if DEPTH
  set_depth(dest, get_depth(src));
#endif
#if SELF_REF
  set_ptr(dest, ref);
#endif
  memcpy(dest->partial, src->partial, min(MAX_PREFIX_LEN, src->partial_len));
}

static void add_child256(art_node256 *n, art_node **ref, unsigned char c,
                         void *child) {
  (void)ref;
  n->n.num_children++;
  n->children[c] = (art_node *)child;
#if SELF_REF
  set_ptr((art_node *)child, &n->children[c]);
#endif
}

static void add_child48(art_node48 *n, art_node **ref, unsigned char c,
                        void *child) {
  if (n->n.num_children < 48) {
    int pos = 0;
    while (n->children[pos])
      pos++;
    n->children[pos] = (art_node *)child;
    n->keys[c] = pos + 1;
#if SELF_REF
    if (!IS_LEAF(child)) {
      set_ptr((art_node *)child, &n->children[pos]);
    }
#endif
    n->n.num_children++;
  } else {
    art_node256 *new_node = (art_node256 *)alloc_node(NODE256);
    for (int i = 0; i < 256; i++) {
      if (n->keys[i]) {
        art_node *existing_child = n->children[n->keys[i] - 1];
        new_node->children[i] = existing_child;
#if SELF_REF
        if (!IS_LEAF(existing_child))
          set_ptr(existing_child, &new_node->children[i]);
#endif
      }
    }
    copy_header((art_node *)new_node, (art_node *)n, (void *)ref);
    *ref = (art_node *)new_node;
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node48_cnt--;
#endif
    add_child256(new_node, ref, c, child);
  }
}

static void add_child16(art_node16 *n, art_node **ref, unsigned char c,
                        void *child) {
  if (n->n.num_children < 16) {
    unsigned mask = (1 << n->n.num_children) - 1;

// support non-x86 architectures
#ifdef __i386__
    __m128i cmp;

    // Compare the key to all 16 stored keys
    cmp = _mm_cmplt_epi8(_mm_set1_epi8(c), _mm_loadu_si128((__m128i *)n->keys));

    // Use a mask to ignore children that don't exist
    unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
#ifdef __amd64__
    __m128i cmp;

    // Compare the key to all 16 stored keys
    cmp = _mm_cmplt_epi8(_mm_set1_epi8(c), _mm_loadu_si128((__m128i *)n->keys));

    // Use a mask to ignore children that don't exist
    unsigned bitfield = _mm_movemask_epi8(cmp) & mask;
#else
    // Compare the key to all 16 stored keys
    unsigned bitfield = 0;
    for (short i = 0; i < 16; ++i) {
      if (c < n->keys[i])
        bitfield |= (1 << i);
    }

    // Use a mask to ignore children that don't exist
    bitfield &= mask;
#endif
#endif

    // Check if less than any
    unsigned idx;
    if (bitfield) {
      idx = __builtin_ctz(bitfield);
      memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
      memmove(n->children + idx + 1, n->children + idx,
              (n->n.num_children - idx) * sizeof(void *));
#if SELF_REF
      refresh_self_refs((art_node *)n, idx + 1, n->n.num_children + 1);
#endif
    } else
      idx = n->n.num_children;

    // Set the child
    n->keys[idx] = c;
    n->children[idx] = (art_node *)child;
#if SELF_REF
    if (!IS_LEAF(child)) {
      set_ptr((art_node *)child, &n->children[idx]);
    }
#endif
    n->n.num_children++;
  } else {
    art_node48 *new_node = (art_node48 *)alloc_node(NODE48);

    // Copy the child pointers and populate the key map
    memcpy(new_node->children, n->children, sizeof(void *) * n->n.num_children);
    for (int i = 0; i < n->n.num_children; i++) {
      new_node->keys[n->keys[i]] = i + 1;
#if SELF_REF
      if (!IS_LEAF(new_node->children[i])) {
        set_ptr(new_node->children[i], &new_node->children[i]);
      }
#endif
    }
    copy_header((art_node *)new_node, (art_node *)n, (void *)ref);
    *ref = (art_node *)new_node;
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node16_cnt--;
#endif
    add_child48(new_node, ref, c, child);
  }
}

static void add_child4(art_node4 *n, art_node **ref, unsigned char c,
                       void *child) {
  if (n->n.num_children < 4) {
    int idx;
    for (idx = 0; idx < n->n.num_children; idx++) {
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
    if (!IS_LEAF(child)) {
      set_ptr((art_node *)child, &n->children[idx]);
    }
#endif
  } else {
    art_node16 *new_node = (art_node16 *)alloc_node(NODE16);

    // Copy the child pointers and the key map
    // memcpy(new_node->children, n->children,
    //        sizeof(void *) * n->n.num_children);
    // memcpy(new_node->keys, n->keys,
    //        sizeof(unsigned char) * n->n.num_children);
    for (int i = 0; i < n->n.num_children; i++) {
      new_node->keys[i] = n->keys[i];
      new_node->children[i] = n->children[i];
#if SELF_REF
      if (!IS_LEAF(new_node->children[i])) {
        set_ptr((art_node *)new_node->children[i], &new_node->children[i]);
      }
#endif
    }
    new_node->n.num_children = n->n.num_children;
    copy_header((art_node *)new_node, (art_node *)n, (void *)ref);
    *ref = (art_node *)new_node;
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node4_cnt--;
#endif
    add_child16(new_node, ref, c, child);
  }
}

static void add_child(art_node *n, art_node **ref, unsigned char c,
                      void *child) {
  switch (n->type) {
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
static int prefix_mismatch(const art_node *n, const unsigned char *key,
                           int key_len, int depth) {
  int max_cmp = min(min(MAX_PREFIX_LEN, n->partial_len), key_len - depth);
  int idx;
  for (idx = 0; idx < max_cmp; idx++) {
    if (n->partial[idx] != key[depth + idx])
      return idx;
  }

  // If the prefix is short we can avoid finding a leaf
  if (n->partial_len > MAX_PREFIX_LEN) {
    // Prefix is longer than what we've checked, find a leaf
    art_leaf *l = minimum(n);
    max_cmp = min(l->key_len, key_len) - depth;
    for (; idx < max_cmp; idx++) {
      if (l->key[idx + depth] != key[depth + idx])
        return idx;
    }
  }
  return idx;
}

static void *recursive_insert(art_node *n, art_node **ref,
                              const unsigned char *key, int key_len,
                              void *value, int depth, int *old, int replace,
                              uint16_t logical_depth) {
  // If we are at a NULL node, inject a leaf
  if (!n) {
    // *ref = (art_node *)SET_LEAF(make_leaf(key, key_len, value)); // orig code
    art_leaf *leaf = make_leaf(key, key_len, value);
#if LEAF_CENTRIC
    set_leaf_parent_ptr(leaf, (art_node *)ref);
#endif
    *ref = (art_node *)SET_LEAF(leaf);
    return NULL;
  }

  // If we are at a leaf, we need to replace it with a node
  if (IS_LEAF(n)) {
#if HIT_CNT_TOTAL
    leaf_hit_cnt++;
#endif
    art_leaf *l = LEAF_RAW(n);

    // Check if we are updating an existing value
    if (!leaf_matches(l, key, key_len, depth)) {
      *old = 1;
      void *old_val = l->value;
      if (replace) {
        l->value = value;
#if LEAF_CENTRIC
        increment_leaf_access_count(l);
#endif
      }
      return old_val;
    }

    // New value, we must split the leaf into a node4
    art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
#if DEPTH
    set_depth(&new_node->n, logical_depth);
#endif
    // Create a new leaf
    art_leaf *l2 = make_leaf(key, key_len, value);
#if LEAF_CENTRIC
    set_leaf_parent_ptr(l2, (art_node *)ref);
#endif

    // Determine longest prefix
    int longest_prefix = longest_common_prefix(l, l2, depth);
    new_node->n.partial_len = longest_prefix;
    memcpy(new_node->n.partial, key + depth,
           min(MAX_PREFIX_LEN, longest_prefix));
    // Add the leafs to the new node4
    *ref = (art_node *)new_node;
#if SELF_REF
    set_ptr(&new_node->n, (void *)ref);
#endif
    add_child4(new_node, ref, l->key[depth + longest_prefix], SET_LEAF(l));
    add_child4(new_node, ref, l2->key[depth + longest_prefix], SET_LEAF(l2));
#if HIT_CNT_TOTAL
    switch (n->type) {
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
    return NULL;
  }
#if HIT_CNT
  n->hit_cnt++;
#endif
  // Check if given node has a prefix
  if (n->partial_len) {
    // Determine if the prefixes differ, since we need to split
    int prefix_diff = prefix_mismatch(n, key, key_len, depth);
    if ((uint32_t)prefix_diff >= n->partial_len) {
      depth += n->partial_len;
      goto RECURSE_SEARCH;
    }

    // Create a new node
    art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
    *ref = (art_node *)new_node;
    new_node->n.partial_len = prefix_diff;
    memcpy(new_node->n.partial, n->partial, min(MAX_PREFIX_LEN, prefix_diff));
#if DEPTH
    set_depth(&new_node->n, logical_depth);
    set_depth(n, logical_depth + 1);
#endif
#if SELF_REF
    set_ptr(&new_node->n, (void *)ref);
#endif
    // Adjust the prefix of the old node
    if (n->partial_len <= MAX_PREFIX_LEN) {
      add_child4(new_node, ref, n->partial[prefix_diff], n);
      n->partial_len -= (prefix_diff + 1);
      memmove(n->partial, n->partial + prefix_diff + 1,
              min(MAX_PREFIX_LEN, n->partial_len));
    } else {
      n->partial_len -= (prefix_diff + 1);
      art_leaf *l = minimum(n);
      add_child4(new_node, ref, l->key[depth + prefix_diff], n);
      memcpy(n->partial, l->key + depth + prefix_diff + 1,
             min(MAX_PREFIX_LEN, n->partial_len));
    }

    // Insert the new leaf
    art_leaf *l = make_leaf(key, key_len, value);
#if LEAF_CENTRIC
    set_leaf_parent_ptr(l, (art_node *)ref);
#endif
    add_child4(new_node, ref, key[depth + prefix_diff], SET_LEAF(l));
#if DEPTH
    decrement_depth(n);
    increment_subtree_depth(n); // TODO: is it really necessary to maintain 100%
                                // accurate depth on the way?
#endif
    return NULL;
  }

RECURSE_SEARCH:;

  // Find a child to recurse to
  art_node **child = find_child(n, key[depth]);
  if (child) {
    return recursive_insert(*child, child, key, key_len, value, depth + 1, old,
                            replace, logical_depth + 1);
  }

  // No child, node goes within us
  art_leaf *l = make_leaf(key, key_len, value);
#if LEAF_CENTRIC
  set_leaf_parent_ptr(l, (art_node *)ref);
#endif
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
void *art_insert(art_tree *t, const unsigned char *key, int key_len,
                 void *value) {
#if THREAD
  return art_insert_thread_safe(t, key, key_len, value);
#else
  int old_val = 0;
  void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0,
                               &old_val, 1, 0);
  if (!old_val)
    t->size++;
  return old;
#endif
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
void *art_insert_no_replace(art_tree *t, const unsigned char *key, int key_len,
                            void *value) {
  int old_val = 0;
  void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0,
                               &old_val, 0, 0);
  if (!old_val)
    t->size++;
  return old;
}

static void remove_child256(art_node256 *n, art_node **ref, unsigned char c) {
  n->children[c] = NULL;
  n->n.num_children--;

  // Resize to a node48 on underflow, not immediately to prevent
  // trashing if we sit on the 48/49 boundary
  if (n->n.num_children == 37) {
    art_node48 *new_node = (art_node48 *)alloc_node(NODE48);
    *ref = (art_node *)new_node;
    copy_header((art_node *)new_node, (art_node *)n, (void *)ref);

    int pos = 0;
    for (int i = 0; i < 256; i++) {
      if (n->children[i]) {
        art_node *child = n->children[i];
        new_node->children[pos] = child;
        new_node->keys[i] = pos + 1;
#if SELF_REF
        if (!IS_LEAF(child))
          set_ptr(child, &new_node->children[pos]);
#endif
        pos++;
      }
    }
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node256_cnt--;
#endif
  }
}

static void remove_child48(art_node48 *n, art_node **ref, unsigned char c) {
  int pos = n->keys[c];
  n->keys[c] = 0;
  n->children[pos - 1] = NULL;
  n->n.num_children--;

  if (n->n.num_children == 12) {
    art_node16 *new_node = (art_node16 *)alloc_node(NODE16);
    *ref = (art_node *)new_node;
    copy_header((art_node *)new_node, (art_node *)n, (void *)ref);

    int child = 0;
    for (int i = 0; i < 256; i++) {
      pos = n->keys[i];
      if (pos) {
        new_node->keys[child] = i;
        new_node->children[child] = n->children[pos - 1];
#if SELF_REF
        if (!IS_LEAF(new_node->children[child]))
          set_ptr(new_node->children[child], &new_node->children[child]);
#endif
        child++;
      }
    }
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node48_cnt--;
#endif
  }
}

static void remove_child16(art_node16 *n, art_node **ref, art_node **l) {
  int pos = l - n->children;
  memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - 1 - pos);
  memmove(n->children + pos, n->children + pos + 1,
          (n->n.num_children - 1 - pos) * sizeof(void *));
#if SELF_REF
  refresh_self_refs((art_node *)n, pos, n->n.num_children);
#endif
  n->n.num_children--;

  if (n->n.num_children == 3) {
    art_node4 *new_node = (art_node4 *)alloc_node(NODE4);
    *ref = (art_node *)new_node;
    copy_header((art_node *)new_node, (art_node *)n, (void *)ref);
    memcpy(new_node->keys, n->keys, 4);
    memcpy(new_node->children, n->children, 4 * sizeof(void *));
#if SELF_REF
    for (int i = 0; i < 4; i++) {
      art_node *child = new_node->children[i];
      if (!IS_LEAF(child))
        set_ptr(child, &new_node->children[i]);
    }
#endif
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node16_cnt--;
#endif
  }
}

static void remove_child4(art_node4 *n, art_node **ref, art_node **l) {
  int pos = l - n->children;
  memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - 1 - pos);
  memmove(n->children + pos, n->children + pos + 1,
          (n->n.num_children - 1 - pos) * sizeof(void *));
#if SELF_REF
  refresh_self_refs((art_node *)n, pos, n->n.num_children);
#endif
  n->n.num_children--;

  // Remove nodes with only a single child
  if (n->n.num_children == 1) {
    art_node *child = n->children[0];
    if (!IS_LEAF(child)) {
      // Concatenate the prefixes
      int prefix = n->n.partial_len;
      if (prefix < MAX_PREFIX_LEN) {
        n->n.partial[prefix] = n->keys[0];
        prefix++;
      }
      if (prefix < MAX_PREFIX_LEN) {
        int sub_prefix = min(child->partial_len, MAX_PREFIX_LEN - prefix);
        memcpy(n->n.partial + prefix, child->partial, sub_prefix);
        prefix += sub_prefix;
      }

      // Store the prefix in the child
      memcpy(child->partial, n->n.partial, min(prefix, MAX_PREFIX_LEN));
      child->partial_len += n->n.partial_len + 1;
    }
    *ref = child;
#if DEPTH
    set_depth(child, get_depth(&n->n));
#endif
#if SELF_REF
    if (!IS_LEAF(child))
      set_ptr(child, (void *)ref);
#endif
#if CUS_ALLOC
    free_node_auto((void *)n);
#else
    free(n);
#endif
#if CNT
    node4_cnt--;
#endif
  }
}

static void remove_child(art_node *n, art_node **ref, unsigned char c,
                         art_node **l) {
  switch (n->type) {
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

static art_leaf *recursive_delete(art_node *n, art_node **ref,
                                  const unsigned char *key, int key_len,
                                  int depth) {
  // Search terminated
  if (!n)
    return NULL;

  // Handle hitting a leaf node
  if (IS_LEAF(n)) {
#if HIT_CNT_TOTAL
    leaf_hit_cnt++;
#endif
    art_leaf *l = LEAF_RAW(n);
    if (!leaf_matches(l, key, key_len, depth)) {
      *ref = NULL;
      return l;
    }
    return NULL;
  }
#if HIT_CNT
  n->hit_cnt++;
#endif
#if HIT_CNT_TOTAL
  switch (n->type) {
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
  // Bail if the prefix does not match
  if (n->partial_len) {
    int prefix_len = check_prefix(n, key, key_len, depth);
    if (prefix_len != min(MAX_PREFIX_LEN, n->partial_len)) {
      return NULL;
    }
    depth = depth + n->partial_len;
  }

  // Find child node
  art_node **child = find_child(n, key[depth]);
  if (!child)
    return NULL;

  // If the child is leaf, delete from this node
  if (IS_LEAF(*child)) {
    art_leaf *l = LEAF_RAW(*child);
    if (!leaf_matches(l, key, key_len, depth)) {
      remove_child(n, ref, key[depth], child);
      return l;
    }
    return NULL;

    // Recurse
  } else {
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
void *art_delete(art_tree *t, const unsigned char *key, int key_len) {
#if THREAD
  return art_delete_thread_safe(t, key, key_len);
#else
  art_leaf *l = recursive_delete(t->root, &t->root, key, key_len, 0);
  if (l) {
    t->size--;
    void *old = l->value;
    free(l);
#if CNT
    leaf_cnt--;
#endif
    return old;
  }
  return NULL;
#endif
}

// Recursively iterates over the tree
static int recursive_iter(art_node *n, art_callback cb, void *data) {
  // Handle base cases
  if (!n)
    return 0;
  if (IS_LEAF(n)) {
#if HIT_CNT_TOTAL
    leaf_hit_cnt++;
#endif
    art_leaf *l = LEAF_RAW(n);
    return cb(data, (const unsigned char *)l->key, l->key_len, l->value);
  }
#if HIT_CNT_TOTAL
  switch (n->type) {
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
#if HIT_CNT
  n->hit_cnt++;
#endif
  int idx, res;
  switch (n->type) {
  case NODE4:
    for (int i = 0; i < n->num_children; i++) {
      res = recursive_iter(((art_node4 *)n)->children[i], cb, data);
      if (res)
        return res;
    }
    break;

  case NODE16:
    for (int i = 0; i < n->num_children; i++) {
      res = recursive_iter(((art_node16 *)n)->children[i], cb, data);
      if (res)
        return res;
    }
    break;

  case NODE48:
    for (int i = 0; i < 256; i++) {
      idx = ((art_node48 *)n)->keys[i];
      if (!idx)
        continue;

      res = recursive_iter(((art_node48 *)n)->children[idx - 1], cb, data);
      if (res)
        return res;
    }
    break;

  case NODE256:
    for (int i = 0; i < 256; i++) {
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
int art_iter(art_tree *t, art_callback cb, void *data) {
  return recursive_iter(t->root, cb, data);
}

/**
 * Checks if a leaf prefix matches
 * @return 0 on success.
 */
static int leaf_prefix_matches(const art_leaf *n, const unsigned char *prefix,
                               int prefix_len) {
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
int art_iter_prefix(art_tree *t, const unsigned char *key, int key_len,
                    art_callback cb, void *data) {
  art_node **child;
  art_node *n = t->root;
  int prefix_len, depth = 0;
  while (n) {
    // Might be a leaf
    if (IS_LEAF(n)) {
      n = (art_node *)LEAF_RAW(n);
      // Check if the expanded path matches
      if (!leaf_prefix_matches((art_leaf *)n, key, key_len)) {
        art_leaf *l = (art_leaf *)n;
        return cb(data, (const unsigned char *)l->key, l->key_len, l->value);
      }
      return 0;
    }

    // If the depth matches the prefix, we need to handle this node
    if (depth == key_len) {
      art_leaf *l = minimum(n);
      if (!leaf_prefix_matches(l, key, key_len))
        return recursive_iter(n, cb, data);
      return 0;
    }

    // Bail if the prefix does not match
    if (n->partial_len) {
      prefix_len = prefix_mismatch(n, key, key_len, depth);

      // Guard if the mis-match is longer than the MAX_PREFIX_LEN
      if ((uint32_t)prefix_len > n->partial_len) {
        prefix_len = n->partial_len;
      }

      // If there is no match, search is terminated
      if (!prefix_len) {
        return 0;

        // If we've matched the prefix, iterate on this node
      } else if (depth + prefix_len == key_len) {
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

#if DEPTH
static void increment_subtree_depth(art_node *n) {
  if (!n)
    return;

  if (IS_LEAF(n)) {
    return;
  }
  // total_subtree_incremented_nodes++;
  increment_depth(n);

  switch (n->type) {
  case NODE4: {
    art_node4 *node = (art_node4 *)n;
    for (int i = 0; i < node->n.num_children; i++)
      increment_subtree_depth(node->children[i]);
#if STATIC
    try_copy_node(n);
#endif
    break;
  }
  case NODE16: {
    art_node16 *node = (art_node16 *)n;
    for (int i = 0; i < node->n.num_children; i++)
      increment_subtree_depth(node->children[i]);
#if STATIC
    try_copy_node(n);
#endif
    break;
  }
  case NODE48: {
    art_node48 *node = (art_node48 *)n;
    for (int i = 0; i < 256; i++) {
      if (node->keys[i]) {
        int idx = node->keys[i] - 1;
        increment_subtree_depth(node->children[idx]);
      }
    }
#if STATIC
    try_copy_node(n);
#endif
    break;
  }
  case NODE256: {
    art_node256 *node = (art_node256 *)n;
    for (int i = 0; i < 256; i++) {
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

void collect_node_depths(art_node *n, int depth, FILE *fd) {
  if (!n)
    return;
  if (IS_LEAF(n)) {
    //         fprintf(fd, "leaf: %d", depth);
    //         art_leaf *leaf = LEAF_RAW(n);
    // #if DEPTH_INDI
    //         fprintf(fd, " %d", leaf->depth);
    // #endif
    //         fprintf(fd, " %s", leaf->key);
    //         fprintf(fd, " \n");
    return;
  }
  switch (n->type) {
  case NODE4: {
    fprintf(fd, "4: %p %d", n, depth);
    fprintf(fd, " %d", get_depth(n));
    fprintf(fd, " \n");
    art_node4 *node = (art_node4 *)n;
    for (int i = 0; i < node->n.num_children; i++) {
      collect_node_depths(node->children[i], depth + 1, fd);
    }
    break;
  }

  case NODE16: {
    fprintf(fd, "16: %p %d", n, depth);
    fprintf(fd, " %d", get_depth(n));
    fprintf(fd, " \n");
    art_node16 *node = (art_node16 *)n;
    for (int i = 0; i < node->n.num_children; i++) {
      collect_node_depths(node->children[i], depth + 1, fd);
    }
    break;
  }

  case NODE48: {
    fprintf(fd, "48: %p %d", n, depth);
    fprintf(fd, " %d", get_depth(n));
    fprintf(fd, " \n");
    art_node48 *node = (art_node48 *)n;
    for (int i = 0; i < 256; i++) {
      uint8_t idx = node->keys[i];
      if (idx) {
        collect_node_depths(node->children[idx - 1], depth + 1, fd);
      }
    }
    break;
  }

  case NODE256: {
    fprintf(fd, "256: %p %d", n, depth);
    fprintf(fd, " %d", get_depth(n));
    fprintf(fd, " \n");
    art_node256 *node = (art_node256 *)n;
    for (int i = 0; i < 256; i++) {
      if (node->children[i]) {
        collect_node_depths(node->children[i], depth + 1, fd);
      }
    }
    break;
  }

  default:
    abort();
  }
}
#endif
#if SELF_REF
static inline void refresh_self_refs(art_node *n, int start, int end) {
  for (int i = start; i < end; ++i) {
    switch (n->type) {
    case NODE4: {
      art_node4 *n4 = (art_node4 *)n;
      art_node *child = n4->children[i];
      if (!IS_LEAF(child))
        set_ptr(child, &n4->children[i]);
      break;
    }
    case NODE16: {
      art_node16 *n16 = (art_node16 *)n;
      art_node *child = n16->children[i];
      if (!IS_LEAF(child))
        set_ptr(child, &n16->children[i]);
      break;
    }
    default:
      abort();
    }
  }
}
static void fix_children_self_ref(void **children, int count) {
  // for (int i = 0; i < count; ++i)
  // {
  //     if (!IS_LEAF(children[i]))
  //     {
  //         ((art_node *)children[i])->self_ref = (art_node **)&children[i];
  //     }
  // }
}
void dump_self_ref_json(FILE *out, art_node *n, void *parent_child_ptr) {
  if (!n) {
    fprintf(out, "null");
    return;
  }

  if (IS_LEAF(n))

  {

    // You can extend this to emit leaf-specific info if needed

    fprintf(out, "{ \"addr\": \"%p\", \"type\": \"leaf\"", (void *)n);

    // #if LEAF_REF
    //         // fprintf(out, ", \"parent_child_ptr\": \"%p\"\n",
    //         parent_child_ptr); fprintf(out, ", \"parent_child_ptr\":
    //         \"%p\"\n", parent_child_ptr); fprintf(out, ", \"parent\":
    //         \"%p\"}\n", LEAF_RAW(n)->self_ref);
    // #else

    fprintf(out, " }\n");

    // #endif

    return;
  }

  fprintf(out, "{\n");
  fprintf(out, "  \"parent_child_ptr\": \"%p\",\n", parent_child_ptr);
  fprintf(out, "  \"addr\": \"%p\",\n", (void *)n);
  fprintf(out, "  \"type\": %d,\n", n->type);
  fprintf(out, "  \"parent\": \"%p\",\n", get_ptr(n));
  fprintf(out, "  \"children\": [\n");

  bool first = true;
  switch (n->type) {
  case NODE4: {
    art_node4 *node = (art_node4 *)n;
    for (int i = 0; i < node->n.num_children; i++) {
      if (!first)
        fprintf(out, ",\n");
      first = false;
      dump_self_ref_json(out, node->children[i], &node->children[i]);
    }
    break;
  }
  case NODE16: {
    art_node16 *node = (art_node16 *)n;
    for (int i = 0; i < node->n.num_children; i++) {
      if (!first)
        fprintf(out, ",\n");
      first = false;
      dump_self_ref_json(out, node->children[i], &node->children[i]);
    }
    break;
  }
  case NODE48: {
    art_node48 *node = (art_node48 *)n;
    for (int i = 0; i < 256; i++) {
      if (node->keys[i]) {
        int idx = node->keys[i] - 1;
        if (!first)
          fprintf(out, ",\n");
        first = false;
        dump_self_ref_json(out, node->children[idx], &node->children[idx]);
      }
    }
    break;
  }
  case NODE256: {
    art_node256 *node = (art_node256 *)n;
    for (int i = 0; i < 256; i++) {
      if (node->children[i]) {
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
#if HIT_CNT_TOTAL
void node_hit_cnt_total() {
  printf("------- Node Hit Count Summry -------\n");
  printf("Node4: %lu\n", node4_hit_cnt);
  printf("Node16: %lu\n", node16_hit_cnt);
  printf("Node48: %lu\n", node48_hit_cnt);
  printf("Node256: %lu\n", node256_hit_cnt);
  printf("Leaf: %lu\n", leaf_hit_cnt);
}
void reset_node_hit_cnt_total() {
  printf("------- Resetting node hit count total -------\n");
  node4_hit_cnt = node16_hit_cnt = node48_hit_cnt = node256_hit_cnt =
      leaf_hit_cnt = 0;
}
#endif

#if CNT
void node_cnt_stat() {
  printf("------- Node Count Summry -------\n");
  float node4_mb = node4_cnt * sizeof(art_node4) / (1024 * 1024);
  float node16_mb = node16_cnt * sizeof(art_node16) / (1024 * 1024);
  float node48_mb = node48_cnt * sizeof(art_node48) / (1024 * 1024);
  float node256_mb = node256_cnt * sizeof(art_node256) / (1024 * 1024);
  float total_mb = node4_mb + node16_mb + node48_mb + node256_mb;

  printf("Node4: %lu, %.2f MB, %.2f pages\n", node4_cnt, node4_mb,
         ceil(node4_mb * 256));
  printf("Node16: %lu, %.2f MB, %.2f pages\n", node16_cnt, node16_mb,
         ceil(node16_mb * 256));
  printf("Node48: %lu, %.2f MB, %.2f pages\n", node48_cnt, node48_mb,
         ceil(node48_mb * 256));
  printf("Node256: %lu, %.2f MB, %.2f pages\n", node256_cnt, node256_mb,
         ceil(node256_mb * 256));
  printf("total: %.2f MB\n", total_mb);
  printf("Leaf: %lu\n", leaf_cnt);
}
#endif
#if LEAF_CUS_ALLOC
// Leaf size class thresholds (in bytes)
static const size_t leaf_size_thresholds[LEAF_SIZE_CLASSES] = {
    16, 24, 32, 40, 48, 54, 60, 72, SIZE_MAX};

// Get the appropriate size class for a leaf
static int get_leaf_size_class(size_t total_size) {
  for (int i = 0; i < LEAF_SIZE_CLASSES; i++) {
    if (total_size <= leaf_size_thresholds[i]) {
      return i;
    }
  }
  return LEAF_SIZE_CLASSES - 1; // Use largest class
}

// Get the allocator for a specific size class
static node_allocator *get_leaf_allocator(int size_class) {
  switch (size_class) {
  case 0:
    return &na_leaf_16;
  case 1:
    return &na_leaf_24;
  case 2:
    return &na_leaf_32;
  case 3:
    return &na_leaf_40;
  case 4:
    return &na_leaf_48;
  case 5:
    return &na_leaf_54;
  case 6:
    return &na_leaf_60;
  case 7:
    return &na_leaf_large;
  default:
    printf("too large size class: %d\n", size_class);
    abort();
  }
}

// Find which allocator a leaf belongs to
static node_allocator *find_leaf_allocator(art_leaf *leaf) {
  for (int i = 0; i < LEAF_SIZE_CLASSES; i++) {
    node_allocator *na = get_leaf_allocator(i);
    uintptr_t addr = (uintptr_t)leaf;
    uintptr_t base = (uintptr_t)na->base_addr;
    uintptr_t end = base + na->node_size * na->capacity;
    if (addr >= base && addr < end) {
      return na;
    }
  }
  return NULL;
}
#endif

#if THREAD
// Global lock table
lock_table_t *global_lock_table = NULL;
background_worker_t *global_worker = NULL;

// Internal search function (existing art_search logic) - REMOVED, using
// art_search_with_version_check instead

// Pessimistic read with locks (fallback) - REMOVED DUPLICATE
// Clean optimistic search with root-only versioning (fast path)
void *art_search_optimistic(const art_tree *t, const unsigned char *key,
                            int key_len) {
  uint32_t version_before, version_after;
  void *result;
  int retries = 0;
  const int max_retries = 10;

  do {
    // Root-only versioning - much faster!
    version_before = t->root ? FAST_GET_VERSION(t->root) : 0;
    result = art_search_internal(t, key, key_len);
    version_after = t->root ? FAST_GET_VERSION(t->root) : 0;

    if (version_before != version_after) {
      retries++;
      continue;
    }
    return result;
  } while (retries < max_retries);

  // Fall back to pessimistic read if too many retries
  return art_search_pessimistic(t, key, key_len);
}

// Internal search function (no version checking - fast)
void *art_search_internal(const art_tree *t, const unsigned char *key,
                          int key_len) {
  art_node **child;
  art_node *n = t->root;
  int prefix_len, depth = 0;

  while (n) {
    // Might be a leaf
    if (IS_LEAF(n)) {
#if HIT_CNT_TOTAL
      leaf_hit_cnt++;
#endif
      n = (art_node *)LEAF_RAW(n);
      // Check if the expanded path matches
      if (!leaf_matches((art_leaf *)n, key, key_len, depth)) {
#if LEAF_CENTRIC
        increment_leaf_access_count((art_leaf *)n);
#endif
        return ((art_leaf *)n)->value;
      }
      return NULL;
    }
#if HIT_CNT
    n->hit_cnt++;
#endif
#if HIT_CNT_TOTAL
    switch (n->type) {
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

    // Bail if the prefix does not match
    if (n->partial_len) {
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

// Pessimistic read with locks (fallback)
static void *art_search_pessimistic(const art_tree *t, const unsigned char *key,
                                    int key_len) {
  pthread_rwlock_rdlock((pthread_rwlock_t *)&t->tree_lock);
  void *result = art_search_internal(t, key, key_len);
  pthread_rwlock_unlock((pthread_rwlock_t *)&t->tree_lock);
  return result;
}

// Migration detection function - REMOVED (unused dead code)
// The current implementation uses root-only versioning for all operations
// and relies on the migration status bit in the version field for migration
// detection

// Hash function for lock table
static size_t hash_node_ptr(art_node *node) {
  return (size_t)node % 1024; // Simple hash, adjust size as needed
}

// Create lock table
lock_table_t *create_lock_table(size_t size) {
  lock_table_t *table = malloc(sizeof(lock_table_t));
  if (!table)
    return NULL;

  table->size = size;
  table->buckets = calloc(size, sizeof(lock_entry_t *));
  if (!table->buckets) {
    free(table);
    return NULL;
  }

  pthread_mutex_init(&table->table_lock, NULL);
  return table;
}

// Destroy lock table
void destroy_lock_table(lock_table_t *table) {
  if (!table)
    return;

  pthread_mutex_lock(&table->table_lock);

  for (size_t i = 0; i < table->size; i++) {
    lock_entry_t *entry = table->buckets[i];
    while (entry) {
      lock_entry_t *next = entry->next;
      pthread_rwlock_destroy(&entry->lock_info.node_lock);
      pthread_rwlock_destroy(&entry->lock_info.parent_lock);
      free(entry);
      entry = next;
    }
  }

  pthread_mutex_unlock(&table->table_lock);
  pthread_mutex_destroy(&table->table_lock);
  free(table->buckets);
  free(table);
}

// Get or create lock info for a node
art_node_lock_t *get_node_lock_info(art_node *node, lock_table_t *table) {
  if (!node || !table)
    return NULL;

  size_t hash = hash_node_ptr(node);
  size_t bucket = hash % table->size;

  pthread_mutex_lock(&table->table_lock);

  // Search for existing entry
  lock_entry_t *entry = table->buckets[bucket];
  while (entry) {
    if (entry->node == node) {
      pthread_mutex_unlock(&table->table_lock);
      return &entry->lock_info;
    }
    entry = entry->next;
  }

  // Create new entry
  entry = malloc(sizeof(lock_entry_t));
  if (!entry) {
    pthread_mutex_unlock(&table->table_lock);
    return NULL;
  }

  entry->node = node;
  entry->lock_info.parent = NULL; // Will be set when needed
  pthread_rwlock_init(&entry->lock_info.node_lock, NULL);
  pthread_rwlock_init(&entry->lock_info.parent_lock, NULL);

  // Insert at head of bucket
  entry->next = table->buckets[bucket];
  table->buckets[bucket] = entry;

  pthread_mutex_unlock(&table->table_lock);
  return &entry->lock_info;
}

// Clean up locks for a node (called when node is destroyed)
void cleanup_node_locks(art_node *node, lock_table_t *table) {
  if (!node || !table)
    return;

  size_t hash = hash_node_ptr(node);
  size_t bucket = hash % table->size;

  pthread_mutex_lock(&table->table_lock);

  lock_entry_t **prev = &table->buckets[bucket];
  lock_entry_t *entry = table->buckets[bucket];

  while (entry) {
    if (entry->node == node) {
      *prev = entry->next;
      pthread_rwlock_destroy(&entry->lock_info.node_lock);
      pthread_rwlock_destroy(&entry->lock_info.parent_lock);
      free(entry);
      break;
    }
    prev = &entry->next;
    entry = entry->next;
  }

  pthread_mutex_unlock(&table->table_lock);
}

// Background worker thread function
void *background_worker_thread(void *arg) {
  background_worker_t *worker = (background_worker_t *)arg;

  while (1) {
    pthread_mutex_lock(&worker->worker_mutex);

    // Wait for condition or check if should stop
    while (!worker->should_stop) {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 1; // Sleep for 1 second

      int ret = pthread_cond_timedwait(&worker->worker_cond,
                                       &worker->worker_mutex, &ts);
      if (ret == ETIMEDOUT) {
        // Time to run sampling
        break;
      }
    }

    if (worker->should_stop) {
      pthread_mutex_unlock(&worker->worker_mutex);
      break;
    }

    pthread_mutex_unlock(&worker->worker_mutex);

    // Run sampling function
    sampling(worker->tree, worker->lock_table);
  }

  return NULL;
}

// Start background worker
background_worker_t *start_background_worker(art_tree *tree,
                                             lock_table_t *lock_table) {
  background_worker_t *worker = malloc(sizeof(background_worker_t));
  if (!worker)
    return NULL;

  worker->tree = tree;
  worker->lock_table = lock_table;
  worker->should_stop = false;

  pthread_mutex_init(&worker->worker_mutex, NULL);
  pthread_cond_init(&worker->worker_cond, NULL);

  if (pthread_create(&worker->worker_thread, NULL, background_worker_thread,
                     worker) != 0) {
    pthread_mutex_destroy(&worker->worker_mutex);
    pthread_cond_destroy(&worker->worker_cond);
    free(worker);
    return NULL;
  }

  return worker;
}

// Stop background worker
void stop_background_worker(background_worker_t *worker) {
  if (!worker)
    return;

  pthread_mutex_lock(&worker->worker_mutex);
  worker->should_stop = true;
  pthread_cond_signal(&worker->worker_cond);
  pthread_mutex_unlock(&worker->worker_mutex);

  pthread_join(worker->worker_thread, NULL);

  pthread_mutex_destroy(&worker->worker_mutex);
  pthread_cond_destroy(&worker->worker_cond);
  free(worker);
}

// Sampling function with atomic operations for background workers
void sampling(art_tree *tree, lock_table_t *lock_table) {
  // Background workers use atomic operations (e.g., CAS) to access shared state
  static uint64_t migration_count = 0;
  static uint64_t node_count = 0;

  // Atomic increment of counters
  __sync_fetch_and_add(&migration_count, 1);
  __sync_fetch_and_add(&node_count, 1);

  // Update total leaf count in histogram
  global_histogram.total_leaves = leaf_cnt;

// Traverse tree to identify hot paths and cold nodes
#if HISTOGRAM
  identify_migration_candidates(tree);

  // Perform actual migrations for identified candidates
  perform_migrations(tree, lock_table);
#endif

  printf("doing sampling - migration_count: %lu, node_count: %lu\n",
         migration_count, node_count);
  printf(
      "histogram - hot_threshold: %u, cold_threshold: %u, total_leaves: %u\n",
      global_histogram.hot_threshold, global_histogram.cold_threshold,
      global_histogram.total_leaves);
}

int migrate_node(art_tree *t, art_node *node, art_node *parent) {
  art_node_lock_t *lock_info = get_node_lock_info(node, global_lock_table);
  if (!lock_info)
    return -1;

  // Lock both node and parent (per-node locking for migrations)
  pthread_rwlock_wrlock(&lock_info->node_lock);
  if (parent) {
    pthread_rwlock_wrlock(&lock_info->parent_lock);
  }

  // Set migration status and increment per-node version (migration-specific)
  node->version ^= MIGRATION_STATUS_BIT; // Flip migration bit
  FAST_INCREMENT_VERSION(node);          // Increment per-node version

  // Perform migration logic here
  // ... migration implementation ...

  // Increment per-node version and clear migration status
  FAST_INCREMENT_VERSION(node);          // Increment per-node version
  node->version ^= MIGRATION_STATUS_BIT; // Flip migration bit back

  // Unlock in reverse order
  if (parent) {
    pthread_rwlock_unlock(&lock_info->parent_lock);
  }
  pthread_rwlock_unlock(&lock_info->node_lock);

  return 0;
}

// Thread-safe insert operation with root-only versioning (consistent with
// reads)
void *art_insert_thread_safe(art_tree *t, const unsigned char *key, int key_len,
                             void *value) {
  pthread_rwlock_wrlock(&t->tree_lock);

  // Increment root version before modification (consistent with read
  // versioning)
  if (t->root) {
    FAST_INCREMENT_VERSION(t->root);
  }

  int old_val = 0;
  void *old = recursive_insert(t->root, &t->root, key, key_len, value, 0,
                               &old_val, 1, 0);
  if (!old_val)
    t->size++;

  // Increment root version after modification (consistent with read versioning)
  if (t->root) {
    FAST_INCREMENT_VERSION(t->root);
  }

  pthread_rwlock_unlock(&t->tree_lock);
  return old;
}

// Thread-safe delete operation with root-only versioning (consistent with
// reads)
void *art_delete_thread_safe(art_tree *t, const unsigned char *key,
                             int key_len) {
  pthread_rwlock_wrlock(&t->tree_lock);

  // Increment root version before modification (consistent with read
  // versioning)
  if (t->root) {
    FAST_INCREMENT_VERSION(t->root);
  }

  art_leaf *l = recursive_delete(t->root, &t->root, key, key_len, 0);
  if (l) {
    t->size--;
    void *old = l->value;
    free(l);
#if CNT
    leaf_cnt--;
#endif
    // Increment root version after modification (consistent with read
    // versioning)
    if (t->root) {
      FAST_INCREMENT_VERSION(t->root);
    }
    pthread_rwlock_unlock(&t->tree_lock);
    return old;
  }

  // Increment root version after modification (even if no deletion)
  if (t->root) {
    FAST_INCREMENT_VERSION(t->root);
  }
  pthread_rwlock_unlock(&t->tree_lock);
  return NULL;
}

// Thread-safe search operation
void *art_search_thread_safe(const art_tree *t, const unsigned char *key,
                             int key_len) {
  return art_search_optimistic(t, key, key_len);
}

#endif // THREAD

#if HISTOGRAM
// Global histogram
access_histogram_t global_histogram;

// Initialize the access histogram
void init_access_histogram(void) {
  memset(&global_histogram, 0, sizeof(access_histogram_t));
  global_histogram.p_hot = DEFAULT_P_HOT;
  global_histogram.p_cold = DEFAULT_P_COLD;
  global_histogram.hot_threshold = 1;
  global_histogram.cold_threshold = 1;
}

// Get the histogram bin for a given frequency (logarithmic scale)
uint32_t get_frequency_bin(uint32_t frequency) {
  if (frequency == 0)
    return 0;

  // Logarithmic binning: bin = log2(frequency)
  uint32_t bin = 0;
  uint32_t freq = frequency;
  while (freq > 1 && bin < HISTOGRAM_BINS - 1) {
    freq >>= 1;
    bin++;
  }
  return bin;
}

// Update leaf access frequency in histogram by traversing all leaf memory regions
void update_leaf_access_frequency(art_leaf *leaf) {
  // Reset histogram bins
  memset(global_histogram.bins, 0, sizeof(global_histogram.bins));
  global_histogram.total_leaves = 0;

  // Traverse all 8 leaf memory regions
  node_allocator *leaf_allocators[] = {
    &na_leaf_16,    // 9-16 bytes
    &na_leaf_24,    // 17-24 bytes
    &na_leaf_32,    // 25-32 bytes
    &na_leaf_40,    // 33-40 bytes
    &na_leaf_48,    // 41-48 bytes
    &na_leaf_54,    // 49-54 bytes
    &na_leaf_60,    // 55-60 bytes
    &na_leaf_large  // 61-72 bytes
  };

  // Process each leaf allocator
  for (int i = 0; i < 8; i++) {
    node_allocator *allocator = leaf_allocators[i];
    
    // Sequential scan through the allocated region
    for (size_t j = 0; j < allocator->capacity; j++) {
      // Check if this slot is allocated (used)
      if (allocator->bitmap[j] == 1) {
        // Calculate pointer to the leaf object
        art_leaf *leaf_ptr = (art_leaf *)((char *)allocator->base_addr + j * allocator->node_size);
        
        // Get access count for this leaf
        uint32_t access_count = get_leaf_access_count(leaf_ptr);
        
        // Update histogram bin
        uint32_t bin = get_frequency_bin(access_count);
        if (bin < HISTOGRAM_BINS) {
          global_histogram.bins[bin]++;
        }
        
        global_histogram.total_leaves++;
      }
    }
  }

  // Increment operation count for cooling
  global_histogram.operation_count++;

  // Trigger cooling if needed
  if (global_histogram.operation_count >= COOLING_INTERVAL) {
    cool_access_frequencies();
    update_hot_cold_thresholds();
  }
}
// Cooler: halve all leaf access frequencies and shift histogram bins
void cool_access_frequencies(void) {
  // Reset operation count
  global_histogram.operation_count = 0;

  // Shift histogram bins to the left by one position (logarithmic reduction)
  for (int i = 0; i < HISTOGRAM_BINS - 1; i++) {
    global_histogram.bins[i] = global_histogram.bins[i + 1];
  }
  global_histogram.bins[HISTOGRAM_BINS - 1] = 0;

  // Note: Actual leaf frequency halving would require tree traversal
  // This is handled by the background worker during sampling
}
void update_hot_cold_thresholds(void) {
  uint32_t target_hot_count =
      (uint32_t)(global_histogram.total_leaves * global_histogram.p_hot);
  uint32_t target_cold_count =
      (uint32_t)(global_histogram.total_leaves * global_histogram.p_cold);

  uint32_t cumulative_count = 0;

  // Find Thot (from high to low frequency)
  global_histogram.hot_threshold = 0;
  for (int i = HISTOGRAM_BINS - 1; i >= 0; i--) {
    cumulative_count += global_histogram.bins[i];
    if (cumulative_count >= target_hot_count) {
      global_histogram.hot_threshold = (1U << i); // Convert bin to frequency
      break;
    }
  }

  // Find Tcold (from low to high frequency)
  cumulative_count = 0;
  global_histogram.cold_threshold = 0;
  for (int i = 0; i < HISTOGRAM_BINS; i++) {
    cumulative_count += global_histogram.bins[i];
    if (cumulative_count >= target_cold_count) {
      global_histogram.cold_threshold = (1U << i); // Convert bin to frequency
      break;
    }
  }
}
// Check if a leaf is hot (frequency > Thot)
bool is_hot_leaf(art_leaf *leaf) {
  uint32_t freq = get_leaf_access_count(leaf);
  return freq > global_histogram.hot_threshold;
}

// Check if a leaf is cold (frequency < Tcold)
bool is_cold_leaf(art_leaf *leaf) {
  uint32_t freq = get_leaf_access_count(leaf);
  return freq < global_histogram.cold_threshold;
}

// Identify migration candidates based on histogram
static void identify_migration_candidates(art_tree *tree) {
  // This would traverse the tree and identify candidates for migration
  // based on access patterns, node size, etc.

  // For hot paths: identify leaves with frequency > Thot
  // For cold nodes: identify leaves with frequency < Tcold

  // Store candidates for migration in a queue or list
}

// Perform migrations for identified candidates
static void perform_migrations(art_tree *tree, lock_table_t *lock_table) {
  // Process migration candidates
  // Use migrate_node() function for each candidate
}
#endif // HISTOGRAM
