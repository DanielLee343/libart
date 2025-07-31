#include "art.h"
#include <assert.h>
#include <numa.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_KEYS 120000000
#define MAX_OPS 120000001
#define AVG_KEY_LEN 30
#define MAX_LINE_LEN 55
#define DATA_MEM_MASK 1
#define MAX_DEPTH 35
// #define NUM_THREADS 1
#define PER_Q_PERF_OPS 0
#define PER_Q_PERF_POP 0

int num_thread = 1;
#if PER_Q_PERF_OPS || PER_Q_PERF_POP
float *latencies_ops;
float *latencies_pop;
static inline int compare_float(const void *a, const void *b) {
  float fa = *(const float *)a;
  float fb = *(const float *)b;
  return (fa > fb) - (fa < fb);
}
#endif

typedef enum { OP_READ, OP_UPDATE, OP_INSERT } op_t;

typedef struct {
  uint64_t value0;
  uint64_t value1;
  uint64_t value2;
  uint64_t value3;
} value_data;

int pre_load_data(char *keys, int *key_lens, FILE *f);
int preload_ops(char *ops, int *ops_len, op_t *ops_types, FILE *f);
void populate_art(art_tree *tree, char *keys, int *key_lens, int num_keys);
void measure_ops_perf(art_tree *tree, char *ops, int *ops_lens, op_t *ops_types,
                      int num_ops);
void measure_ops_perf_threading(art_tree *tree, char *ops, int *ops_lens,
                                op_t *ops_types, int num_ops);
int print_key_callback(void *data, const unsigned char *key,
                       unsigned int key_len, void *value);

static inline double elapsed_ms(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 +
         (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main(int argc, char *argv[]) {
  // if (argc != 2)
  // {
  //     fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
  //     return 1;
  // }
  if (argc > 2) {
    num_thread = atoi(argv[2]);
    if (num_thread <= 0) {
      fprintf(stderr, "Invalid number of threads: %s\n", argv[2]);
      return 1;
    }
  }
  printf("num_thread = %d\n", num_thread);
  struct timespec t_start, t_end;
  double insert_ms = 0, ops_ms = 0;
  char input_path[128];
  char ops_path[128];
  snprintf(input_path, sizeof(input_path), "/mnt/data_bk/zipfian/load_%s",
           argv[1]);
  //  "/home/lyuze/workspace/ycsb/workloads_zipfian/load_%s", argv[1]);

  snprintf(ops_path, sizeof(ops_path), "/mnt/data_bk/zipfian/txn_%s", argv[1]);
  //  "/home/lyuze/workspace/ycsb/workloads_zipfian/txn_%s", argv[1]);
  srand(42);
  if (numa_available() < 0) {
    fprintf(stderr, "NUMA not available\n");
    return 1;
  }
  // loading input file
  FILE *f_input = fopen(input_path, "r");
  if (!f_input) {
    perror("fopen input file failed");
    exit(EXIT_FAILURE);
  }
  char *keys =
      (char *)numa_alloc_onnode((size_t)MAX_KEYS * AVG_KEY_LEN, DATA_MEM_MASK);
  if (!keys) {
    perror("numa_alloc_onnode for flat keys failed");
    exit(EXIT_FAILURE);
  }

  int *key_lens =
      (int *)numa_alloc_onnode(sizeof(int) * MAX_KEYS, DATA_MEM_MASK);
  if (!key_lens) {
    perror("malloc key_lens failed");
    exit(EXIT_FAILURE);
  }

  int num_keys = pre_load_data(keys, key_lens, f_input);
  fclose(f_input);

  printf("Loaded %d keys\n", num_keys);
#if PER_Q_PERF_OPS
  latencies_ops = malloc(sizeof(float) * num_ops);
#endif
#if PER_Q_PERF_POP
  latencies_pop = malloc(sizeof(float) * num_keys);
#endif

  // loading ops file
  FILE *f_ops = fopen(ops_path, "r");
  if (!f_ops) {
    perror("fopen input file failed");
    exit(EXIT_FAILURE);
  }
  char *ops =
      (char *)numa_alloc_onnode((size_t)MAX_OPS * AVG_KEY_LEN, DATA_MEM_MASK);
  if (!ops) {
    perror("numa_alloc_onnode ops failed");
    exit(EXIT_FAILURE);
  }

  int *ops_lens =
      (int *)numa_alloc_onnode(MAX_OPS * sizeof(int), DATA_MEM_MASK);
  op_t *ops_types =
      (op_t *)numa_alloc_onnode(MAX_OPS * sizeof(op_t), DATA_MEM_MASK);

  if (!ops_lens || !ops_types) {
    perror("malloc ops_len/types failed");
    exit(EXIT_FAILURE);
  }

  int num_ops = preload_ops(ops, ops_lens, ops_types, f_ops);

  fclose(f_ops);

  printf("Loaded %d ops\n", num_ops);

  art_tree t;
  // int res = art_tree_init(&t, argv[1]);
  int res = art_tree_init(&t);
  { // for printing accessed addr
    // char acc_addr_path[64];
    // snprintf(acc_addr_path, sizeof(acc_addr_path),
    //          "access_addr.txt");
    // acc_fd = fopen(acc_addr_path, "w");
  }

  // populate art
  clock_gettime(CLOCK_MONOTONIC, &t_start);
  populate_art(&t, keys, key_lens, num_keys);
  clock_gettime(CLOCK_MONOTONIC, &t_end);
  insert_ms = elapsed_ms(t_start, t_end);
  fprintf(stdout, "insert: %.2f\n", insert_ms / 1000);
  fflush(stdout);

  {
    // reset_node_hit_cnt_total(); // resets for global metadata
    // cooling_node_hit_cnt_individual(t.root, 0); // resets for individual
    // distribute_nodes(t.root, &t.root, 0);
    // print_node_move_stat();
    // start_acc_streaming = 1;
    // stream_node_type_addr(t.root, stdout);
    // node_cnt_stat();
  }

  // measure ops perf
  clock_gettime(CLOCK_MONOTONIC, &t_start);
  // measure_ops_perf(&t, ops, ops_lens, ops_types, num_ops);
  // measure_ops_perf_threading(&t, ops, ops_lens, ops_types, num_ops);
  clock_gettime(CLOCK_MONOTONIC, &t_end);
  ops_ms = elapsed_ms(t_start, t_end);
  fprintf(stdout, "ops: %.2f\n", ops_ms / 1000);
  {
    // node_cnt_stat();
    // char level_stats_path[64];
    // snprintf(level_stats_path, sizeof(level_stats_path),
    //          "zipfian/level_stats_%s.txt", argv[1]);
    // FILE *level_stats_fd = fopen(level_stats_path, "w");
    // stream_level_distribution(t.root, level_stats_fd);
    // fclose(level_stats_fd);
    // node_hit_cnt_total();
    // char hit_cnt_path[64];
    // snprintf(hit_cnt_path, sizeof(hit_cnt_path),
    //          "zipfian/hit_count_%s.txt", argv[1]);
    // FILE *hit_cnt_fd = fopen(hit_cnt_path, "w");
    // stream_node_hit_counts_individual(t.root, hit_cnt_fd);
    // fclose(hit_cnt_fd);

    // FILE *depth_fd = fopen("depth_file.txt", "w"); // stream node type addr
    // and size collect_node_depths(t.root, 0, depth_fd);
    // print_avg_node_depths(&stat);
    // fclose(depth_fd);
    {
      // char hotness_path_file[128];
      // snprintf(hotness_path_file, sizeof(hotness_path_file),
      //          "/mnt/data_bk/hotness_depth_%s.txt", argv[1]);
      // FILE *hotness_path = fopen(hotness_path_file, "w");
      // int hit_cnt_path[MAX_DEPTH];
      // void *node_path[MAX_DEPTH];
      // dfs_print_hit_cnt_path(t.root, 0, hit_cnt_path, node_path,
      // hotness_path); fclose(hotness_path);
    }
  }
  fflush(stdout);
  { // for printing accessed addr
    // fclose(acc_fd);
  }
  {
    // FILE *self_ref_fd = fopen("self_ref.json", "w");
    // fprintf(self_ref_fd, "{ \"tree\": ");
    // dump_self_ref_json(self_ref_fd, t.root, NULL);
    // fprintf(self_ref_fd, " }\n");
    // fclose(self_ref_fd);
  }
  // show_stat();
  // cleaning
  res = art_tree_destroy(&t);
  numa_free(keys, (size_t)MAX_KEYS * AVG_KEY_LEN);
  numa_free(key_lens, sizeof(int) * MAX_KEYS);

  numa_free(ops, (size_t)MAX_OPS * AVG_KEY_LEN);
  numa_free(ops_lens, MAX_OPS * sizeof(int));
  numa_free(ops_types, MAX_OPS * sizeof(op_t));
#if PER_Q_PERF_OPS
  qsort(latencies_ops, num_ops, sizeof(float), compare_float);
  float median = latencies_ops[num_ops / 2];
  float p99 = latencies_ops[(int)(num_ops * 0.99)];
  printf("median: %.3f, p99: %.3f\n", median, p99);
  free(latencies_ops);
#endif
#if PER_Q_PERF_POP
  qsort(latencies_pop, num_keys, sizeof(float), compare_float);
  float median = latencies_pop[num_keys / 2];
  float p99 = latencies_pop[(int)(num_keys * 0.99)];
  printf("median: %.3f, p99: %.3f\n", median, p99);
  free(latencies_pop);
#endif

  return 0;
}

int pre_load_data(char *keys, int *key_lens, FILE *f) {
  int num_keys = 0;
  char buf[MAX_LINE_LEN];
  char *cur = keys;

  while (fgets(buf, sizeof(buf), f)) {
    if (num_keys >= MAX_KEYS) {
      perror("Too small MAX_KEYS");
      abort();
    }

    int len = strlen(buf);
    if (buf[len - 1] == '\n') {
      buf[len - 1] = '\0';
      len--;
    }

    if (strncmp(buf, "INSERT ", 7) != 0)
      continue;

    const char *src = buf + 7;
    len = strlen(src) + 1;
    memcpy(cur, src, len - 1);
    cur[len] = '\0';
    key_lens[num_keys] = len;
    cur += len;
    num_keys++;
  }

  return num_keys;
}
int preload_ops(char *ops, int *ops_len, op_t *ops_types, FILE *f) {
  int num_ops = 0;
  char buf[MAX_LINE_LEN];
  char *cur = ops;

  while (fgets(buf, sizeof(buf), f)) {
    if (num_ops >= MAX_OPS) {
      perror("Too small MAX_OPS");
      abort();
    }

    int len = strlen(buf);
    if (buf[len - 1] == '\n') {
      buf[len - 1] = '\0';
      len--;
    }

    const char *prefix = NULL;
    op_t type;

    if (strncmp(buf, "READ ", 5) == 0) {
      type = OP_READ;
      prefix = buf + 5;
    } else if (strncmp(buf, "UPDATE ", 7) == 0) {
      type = OP_UPDATE;
      prefix = buf + 7;
    } else if (strncmp(buf, "INSERT ", 7) == 0) {
      type = OP_INSERT;
      prefix = buf + 7;
    } else {
      continue; // skip unknown lines
    }
    len = strlen(prefix) + 1;
    memcpy(cur, prefix, len - 1);
    cur[len] = '\0';
    ops_types[num_ops] = type;
    ops_len[num_ops] = len;
    cur += len;
    num_ops++;
  }

  return num_ops;
}

void populate_art(art_tree *tree, char *keys, int *key_lens, int num_keys) {
  size_t offset = 0;
#if PER_Q_PERF_POP
  struct timespec per_q_start, per_q_end;
  double per_q_lat;
#endif

  for (int i = 0; i < num_keys; i++) {
#if PER_Q_PERF_POP
    clock_gettime(CLOCK_MONOTONIC, &per_q_start);
#endif
    const unsigned char *key_ptr = (const unsigned char *)(keys + offset);
    int key_len = key_lens[i];
    art_insert(tree, key_ptr, key_len, (void *)(uintptr_t)(i + 1));

#if PER_Q_PERF_POP
    clock_gettime(CLOCK_MONOTONIC, &per_q_end);
    double per_q_lat = (per_q_end.tv_sec - per_q_start.tv_sec) * 1e6 +
                       (per_q_end.tv_nsec - per_q_start.tv_nsec) / 1e3;
    latencies_pop[i] = per_q_lat;
#endif
    offset += key_lens[i];
    // if (i % 1200000 == 0)
    // {
    //     node_cnt_stat();
    // }
  }
}
void measure_ops_perf(art_tree *tree, char *ops, int *ops_lens, op_t *ops_types,
                      int num_ops) {
  int none_null_cnt = 0;
  size_t offset = 0;
  uintptr_t total_val = 0;
  char hit_cnt_path[128];
  int stream_counter = 0;
#if PER_Q_PERF_OPS
  struct timespec per_q_start, per_q_end;
  double per_q_lat;
#endif
  struct timespec t_start, t_end;
  double sort_ms = 0;
  double sort_time_total = 0.0;
  for (int i = 0; i < num_ops; i++) {
    const unsigned char *ops_ptr = (const unsigned char *)(ops + offset);
    int ops_len = ops_lens[i];
    int ops_type = ops_types[i];
    void *value = (void *)(uintptr_t)i;
#if PER_Q_PERF_OPS
    clock_gettime(CLOCK_MONOTONIC, &per_q_start);
#endif
    if (ops_type == OP_READ) {
      void *val = art_search(tree, ops_ptr, ops_len);
      if (val) {
        // total_val += *(uintptr_t *)val;
        total_val += (uintptr_t)val;
        none_null_cnt++;
      }
    } else if (ops_type == OP_UPDATE || ops_type == OP_INSERT) {
      art_insert(tree, ops_ptr, ops_len, (void *)(uintptr_t)(i + 1));
    }
    // if (i == 0)
    // if (i == 1000000)
    // if (i % 12000000 == 0) // total 10 times
    if (0) {
      // printf("streaming %d...\n", stream_counter);
      // snprintf(hit_cnt_path, sizeof(hit_cnt_path),
      //          "zipfian/email_a_hotness/%d.txt", stream_counter);
      // FILE *hit_cnt_fd = fopen(hit_cnt_path, "w");
      // stream_node_hit_counts_individual(tree->root, hit_cnt_fd);
      // cooling_node_hit_cnt_individual(tree->root, 0.1); // perform cooling
      // fclose(hit_cnt_fd);
      // stream_counter++;
      clock_gettime(CLOCK_MONOTONIC, &t_start);
      // sort_all_hotness(); // old version, do not use
      // get_top_k_and_swap();
      // traverse_tree_populate_min_heap(tree->root);
      clock_gettime(CLOCK_MONOTONIC, &t_end);
      sort_ms = elapsed_ms(t_start, t_end);
      sort_time_total += sort_ms;
      fprintf(stdout, "sort: %.3f\n", sort_ms / 1000);
      // print_min_heap_stat();
      // reset_min_heap();
    }
#if PER_Q_PERF_OPS
    clock_gettime(CLOCK_MONOTONIC, &per_q_end);
    double per_q_lat = (per_q_end.tv_sec - per_q_start.tv_sec) * 1e6 +
                       (per_q_end.tv_nsec - per_q_start.tv_nsec) / 1e3;
    latencies_ops[i] = per_q_lat;
#endif
    offset += ops_len;
  }
  printf("# found: %d, total_val: %ld, sort_time_total: %.3f\n", none_null_cnt,
         total_val, sort_time_total / 1000);
}
int print_key_callback(void *data, const unsigned char *key,
                       unsigned int key_len, void *value) {
  int *counter = (int *)data;
  for (unsigned int i = 0; i < key_len; i++)
    printf("%02x ", key[i]);
  printf("\nkey = \"%.*s\", value = %p\n", key_len, key, value);
  (*counter)++;
  return 0;
}

typedef struct {
  art_tree *tree;
  char *ops;
  int *ops_lens;
  op_t *ops_types;
  int start;
  int end;
  uintptr_t local_total;
  int local_count;
} thread_arg_t;

static void *thread_worker(void *arg) {
  thread_arg_t *targ = (thread_arg_t *)arg;
  int offset = 0;
  for (int i = 0; i < targ->start; i++)
    offset += targ->ops_lens[i];

  printf("%d - %d\n", targ->start, targ->end);
  for (int i = targ->start; i < targ->end; ++i) {
    const unsigned char *ops_ptr = (const unsigned char *)(targ->ops + offset);
    int ops_len = targ->ops_lens[i];
    int ops_type = targ->ops_types[i];

    if (ops_type == OP_READ) {
      void *val = art_search(targ->tree, ops_ptr, ops_len);
      if (val) {
        targ->local_total += *(uintptr_t *)val;
        targ->local_count++;
      }
    }
    offset += ops_len; // advance to next key
  }

  return NULL;
}

void measure_ops_perf_threading(art_tree *tree, char *ops, int *ops_lens,
                                op_t *ops_types, int num_ops) {
  pthread_t threads[num_thread];
  thread_arg_t args[num_thread];

  int chunk = (num_ops + num_thread - 1) / num_thread;
  for (int i = 0; i < num_thread; ++i) {
    args[i].tree = tree;
    args[i].ops = ops;
    args[i].ops_lens = ops_lens;
    args[i].ops_types = ops_types;
    args[i].start = i * chunk;
    args[i].end = (i + 1) * chunk;
    if (args[i].end > num_ops)
      args[i].end = num_ops;
    args[i].local_total = 0;
    args[i].local_count = 0;
    pthread_create(&threads[i], NULL, thread_worker, &args[i]);
  }

  int total_count = 0;
  uintptr_t total_val = 0;
  for (int i = 0; i < num_thread; ++i) {
    pthread_join(threads[i], NULL);
    total_count += args[i].local_count;
    total_val += args[i].local_total;
  }

  printf("# found: %d, total_val: %ld\n", total_count, total_val);
}