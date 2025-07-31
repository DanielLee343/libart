#ifndef NODE_ALLOCATOR_H
#define NODE_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_ALLOCATORS 8
#ifdef __cplusplus
extern "C"
{
#endif
#define CXL_MASK 1
#define LOCAL_MASK 0
    typedef struct
    {
        void *base_addr;    // Base address of mmap region
        size_t node_size;   // Size of each node
        size_t capacity;    // Total number of nodes
        uint8_t *bitmap;    // 1 if used, 0 if free
        size_t used;        // Current used count
        size_t *free_stack; // Stack of free indices
        size_t free_top;    // Top index in the stack
    } node_allocator;

    /**
     * Initializes the allocator with a given capacity and node size.
     * Internally calls mmap to reserve a large chunk.
     * Returns 0 on success, -1 on failure.
     */
    void register_allocator(node_allocator *na);
    void init_allocator(node_allocator *na, size_t capacity, size_t node_size, bool in_cxl);

    /**
     * Allocates a zeroed node within the reserved region.
     * Returns NULL if out of memory.
     */
    void *alloc_node_cus(node_allocator *na);

    /**
     * Frees the given node (marks it as free in the bitmap).
     * The pointer must be within the range of the mmap'd region.
     */
    void free_node(node_allocator *na, void *node_ptr);
    void free_node_auto(void *node_ptr); // scans through existing allocators and performs free

    /**
     * Debug print showing internal allocator stats.
     */
    void debug_allocator(node_allocator *na);

    /**
     * Frees all resources (bitmap, mmap region).
     */
    void destroy_allocator(node_allocator *na);

#ifdef __cplusplus
}
#endif

#endif // NODE_ALLOCATOR_H