#include "art.h"
#include <stdbool.h>

void *leaf_base = NULL;
void *node4_base = NULL;
void *node16_base = NULL;
void *node48_base = NULL;
void *node256_base = NULL;
struct memkind *leaf_kind = NULL;
struct memkind *node4_kind = NULL;
struct memkind *node16_kind = NULL;
struct memkind *node48_kind = NULL;
struct memkind *node256_kind = NULL;

#if STATIC || ONLINE
void *node4_local = NULL;
void *node4_cxl = NULL;
void *node16_local = NULL;
void *node16_cxl = NULL;
void *node48_local = NULL;
void *node48_cxl = NULL;
void *node256_local = NULL;
void *node256_cxl = NULL;
struct memkind *node4_local_kind = NULL;
struct memkind *node4_cxl_kind = NULL;
struct memkind *node16_local_kind = NULL;
struct memkind *node16_cxl_kind = NULL;
struct memkind *node48_local_kind = NULL;
struct memkind *node48_cxl_kind = NULL;
struct memkind *node256_local_kind = NULL;
struct memkind *node256_cxl_kind = NULL;
struct memkind *local_kinds[4];
struct memkind *cxl_kinds[4];
#endif

#if ONLINE
bool node4_local_full = false;
bool node16_local_full = false;
bool node48_local_full = false;
bool node256_local_full = false;

art_node **node4_hot = NULL;
art_node **node16_hot = NULL;
art_node **node48_hot = NULL;
art_node **node256_hot = NULL;
art_node **node4_cold = NULL;
art_node **node16_cold = NULL;
art_node **node48_cold = NULL;
art_node **node256_cold = NULL;

int node4_local_alloc_cnt = 0;
int node4_cxl_alloc_cnt = 0;
int node16_local_alloc_cnt = 0;
int node16_cxl_alloc_cnt = 0;
int node48_local_alloc_cnt = 0;
int node48_cxl_alloc_cnt = 0;
int node256_local_alloc_cnt = 0;
int node256_cxl_alloc_cnt = 0;
int traverse_cnt = 0;
// art_node *LRU_node4[HOT_CACHE_LIMIT]; // used for Node4 LRU
#endif

#if DEPTH_INDI
size_t subtree_inc_func_called = 0;
size_t total_subtree_incremented_nodes = 0;
#endif

#if CNT
unsigned long node4_cnt = 0;
unsigned long node16_cnt = 0;
unsigned long node48_cnt = 0;
unsigned long node256_cnt = 0;
unsigned long leaf_cnt = 0;
#endif

#if ENABLE_PROFILE
int node4_local_cnt = 0;
int node4_cxl_cnt = 0;
int node16_local_cnt = 0;
int node16_cxl_cnt = 0;
int node48_local_cnt = 0;
int node48_cxl_cnt = 0;
int node256_local_cnt = 0;
int node256_cxl_cnt = 0;
int node4_mis_placed = 0;
int node16_mis_placed = 0;
int node48_mis_placed = 0;
int node256_mis_placed = 0;
int *local_cnt[4];
int *cxl_cnt[4];
int *misplaced_cnt[4];
#endif
FILE *log_fd;

#if STATIC
int static_metrics_line_cnt = 0;
int **matrix = NULL;
#endif