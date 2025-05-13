#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numa.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include "art.h"

#define MAX_KEYS 100000000
#define MAX_OPS 120000000
#define AVG_KEY_LEN 20
#define MAX_LINE_LEN 256
#define LOCAL_MASK 0

typedef enum
{
    OP_READ,
    OP_UPDATE,
    OP_INSERT
} op_t;

int pre_load_data(char *keys, int *key_lens, FILE *f);
int preload_ops(char *ops, int *ops_len, op_t *ops_types, FILE *f);
void populate_art(art_tree *tree, char *keys, int *key_lens, int num_keys);
void measure_ops_perf(art_tree *tree, char *ops, int *ops_lens, op_t *ops_types, int num_ops);
int print_key_callback(void *data, const unsigned char *key, unsigned int key_len, void *value);

static double elapsed_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    struct timespec t_start, t_end;
    double insert_ms = 0, ops_ms = 0;
    char input_path[128];
    char ops_path[128];
    snprintf(input_path, sizeof(input_path),
             "/home/lyuze/workspace/ycsb/workloads/load_%s", argv[1]);

    snprintf(ops_path, sizeof(ops_path),
             "/home/lyuze/workspace/ycsb/workloads/txn_%s", argv[1]);
    srand(42);
    if (numa_available() < 0)
    {
        fprintf(stderr, "NUMA not available\n");
        return 1;
    }
    // loading input file
    FILE *f_input = fopen(input_path, "r");
    if (!f_input)
    {
        perror("fopen input file failed");
        exit(EXIT_FAILURE);
    }
    char *keys = (char *)numa_alloc_onnode((size_t)MAX_KEYS * AVG_KEY_LEN, LOCAL_MASK);

    if (!keys)
    {
        perror("numa_alloc_onnode for flat keys failed");
        exit(EXIT_FAILURE);
    }

    int *key_lens = (int *)numa_alloc_onnode(sizeof(int) * MAX_KEYS, LOCAL_MASK);
    if (!key_lens)
    {
        perror("malloc key_lens failed");
        exit(EXIT_FAILURE);
    }

    int num_keys = pre_load_data(keys, key_lens, f_input);

    fclose(f_input);

    printf("Loaded %d keys\n", num_keys);

    FILE *f_ops = fopen(ops_path, "r");
    if (!f_ops)
    {
        perror("fopen input file failed");
        exit(EXIT_FAILURE);
    }
    char *ops = (char *)numa_alloc_onnode((size_t)MAX_OPS * AVG_KEY_LEN, LOCAL_MASK);
    if (!ops)
    {
        perror("numa_alloc_onnode ops failed");
        exit(EXIT_FAILURE);
    }

    int *ops_lens = (int *)numa_alloc_onnode(MAX_OPS * sizeof(int), LOCAL_MASK);
    op_t *ops_types = (op_t *)numa_alloc_onnode(MAX_OPS * sizeof(op_t), LOCAL_MASK);

    if (!ops_lens || !ops_types)
    {
        perror("malloc ops_len/types failed");
        exit(EXIT_FAILURE);
    }

    int num_ops = preload_ops(ops, ops_lens, ops_types, f_ops);

    fclose(f_ops);

    printf("Loaded %d ops\n", num_ops);

    art_tree t;
    int res = art_tree_init(&t);

    // populate art
    size_t off = 0;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    populate_art(&t, keys, key_lens, num_keys);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    insert_ms = elapsed_ms(t_start, t_end);
    fprintf(stdout, "insert: %.2f\n", insert_ms / 1000);
    fflush(stdout);
    node_cnt_stat();

    // measure ops perf
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    measure_ops_perf(&t, ops, ops_lens, ops_types, num_ops);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    ops_ms = elapsed_ms(t_start, t_end);
    fprintf(stdout, "ops_ms: %.2f\n", ops_ms / 1000);
    fflush(stdout);

    // cleaning
    res = art_tree_destroy(&t);
    node_cnt_stat();
    numa_free(keys, (size_t)MAX_KEYS * AVG_KEY_LEN);
    numa_free(key_lens, sizeof(int) * MAX_KEYS);

    numa_free(ops, (size_t)MAX_OPS * AVG_KEY_LEN);
    numa_free(ops_lens, MAX_OPS * sizeof(int));
    numa_free(ops_types, MAX_OPS * sizeof(op_t));
    return 0;
}

int pre_load_data(char *keys, int *key_lens, FILE *f)
{
    int num_keys = 0;
    char buf[MAX_LINE_LEN];
    char *cur = keys;

    while (fgets(buf, sizeof(buf), f))
    {
        if (num_keys >= MAX_KEYS)
        {
            perror("Too small MAX_KEYS");
            abort();
        }

        int len = strlen(buf);
        if (buf[len - 1] == '\n')
        {
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
int preload_ops(char *ops, int *ops_len, op_t *ops_types, FILE *f)
{
    int num_ops = 0;
    char buf[MAX_LINE_LEN];
    char *cur = ops;

    while (fgets(buf, sizeof(buf), f))
    {
        if (num_ops >= MAX_OPS)
        {
            perror("Too small MAX_OPS");
            abort();
        }

        int len = strlen(buf);
        if (buf[len - 1] == '\n')
        {
            buf[len - 1] = '\0';
            len--;
        }

        const char *prefix = NULL;
        op_t type;

        if (strncmp(buf, "READ ", 5) == 0)
        {
            type = OP_READ;
            prefix = buf + 5;
        }
        else if (strncmp(buf, "UPDATE ", 7) == 0)
        {
            type = OP_UPDATE;
            prefix = buf + 7;
        }
        else if (strncmp(buf, "INSERT ", 7) == 0)
        {
            type = OP_INSERT;
            prefix = buf + 7;
        }
        else
        {
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

void populate_art(art_tree *tree, char *keys, int *key_lens, int num_keys)
{
    size_t offset = 0;

    for (int i = 0; i < num_keys; i++)
    {
        const unsigned char *key_ptr = (const unsigned char *)(keys + offset);
        int key_len = key_lens[i];
        void *value = (void *)(uintptr_t)(i + 1);
        art_insert(tree, key_ptr, key_len, value);
        offset += key_lens[i];
    }
}

void measure_ops_perf(art_tree *tree, char *ops, int *ops_lens, op_t *ops_types, int num_ops)
{
    int none_null_cnt = 0;
    size_t offset = 0;
    uintptr_t total_val = 0;

    for (int i = 0; i < num_ops; i++)
    {
        const unsigned char *ops_ptr = (const unsigned char *)(ops + offset);
        int ops_len = ops_lens[i];
        int ops_type = ops_types[i];
        void *value = (void *)(uintptr_t)i;
        if (ops_type == OP_READ)
        {
            void *val = art_search(tree, ops_ptr, ops_len);
            if (val)
            {
                total_val += (uintptr_t)val;
                none_null_cnt++;
            }
        }
        else if (ops_type == OP_UPDATE || ops_type == OP_INSERT)
        {
            art_insert(tree, ops_ptr, ops_len, value);
        }

        offset += ops_len;
    }
    printf("# found: %d, total_val: %ld\n", none_null_cnt, total_val);
}
int print_key_callback(void *data, const unsigned char *key, unsigned int key_len, void *value)
{
    int *counter = (int *)data;
    for (unsigned int i = 0; i < key_len; i++)
        printf("%02x ", key[i]);
    printf("\nkey = \"%.*s\", value = %p\n", key_len, key, value);
    (*counter)++;
    return 0;
}