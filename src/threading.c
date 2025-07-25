
typedef struct
{
    art_tree *tree;
    char *ops;
    int *ops_lens;
    op_t *ops_types;
    int start;
    int end;
    uintptr_t local_total;
    int local_count;
} thread_arg_t;

static void *thread_worker(void *arg)
{
    thread_arg_t *targ = (thread_arg_t *)arg;
    int offset = 0;
    for (int i = 0; i < targ->start; i++)
        offset += targ->ops_lens[i];

    printf("%d - %d\n", targ->start, targ->end);
    for (int i = targ->start; i < targ->end; ++i)
    {
        const unsigned char *ops_ptr = (const unsigned char *)(targ->ops + offset);
        int ops_len = targ->ops_lens[i];
        int ops_type = targ->ops_types[i];

        if (ops_type == OP_READ)
        {
            void *val = art_search(targ->tree, ops_ptr, ops_len);
            if (val)
            {
                targ->local_total += *(uintptr_t *)val;
                targ->local_count++;
            }
        }
        offset += ops_len; // advance to next key
    }

    return NULL;
}

void measure_ops_perf_threading(art_tree *tree, char *ops, int *ops_lens, op_t *ops_types, int num_ops)
{
    pthread_t threads[num_thread];
    thread_arg_t args[num_thread];

    int chunk = (num_ops + num_thread - 1) / num_thread;
    for (int i = 0; i < num_thread; ++i)
    {
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
    for (int i = 0; i < num_thread; ++i)
    {
        pthread_join(threads[i], NULL);
        total_count += args[i].local_count;
        total_val += args[i].local_total;
    }

    printf("# found: %d, total_val: %ld\n", total_count, total_val);
}