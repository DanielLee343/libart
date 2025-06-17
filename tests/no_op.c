#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numa.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <unistd.h>

#define MAX_KEYS 120000000
#define MAX_OPS 120000001
#define AVG_KEY_LEN 30
#define MAX_LINE_LEN 55
#define LOCAL_MASK 0

typedef enum
{
    OP_READ,
    OP_UPDATE,
    OP_INSERT
} op_t;

typedef struct
{
    uint64_t value0;
    uint64_t value1;
    uint64_t value2;
    uint64_t value3;
} value_data;

int pre_load_data(char *keys, int *key_lens, FILE *f);
int preload_ops(char *ops, int *ops_len, op_t *ops_types, FILE *f);

static double elapsed_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

int main(int argc, char *argv[])
{
    // if (argc != 2)
    // {
    //     fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
    //     return 1;
    // }
    struct timespec t_start, t_end;
    double insert_ms = 0, ops_ms = 0;
    char input_path[128];
    char ops_path[128];
    snprintf(input_path, sizeof(input_path),
             "/mnt/data_bk/zipfian/load_%s", argv[1]);
    //  "/home/lyuze/workspace/ycsb/workloads_zipfian/load_%s", argv[1]);

    snprintf(ops_path, sizeof(ops_path),
             "/mnt/data_bk/zipfian/txn_%s", argv[1]);
    //  "/home/lyuze/workspace/ycsb/workloads_zipfian/txn_%s", argv[1]);
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

    // loading ops file
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

    sleep(10);
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