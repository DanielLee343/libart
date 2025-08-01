#include "node_allocator.h"
#include <assert.h>
#include <numaif.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
static node_allocator *all_allocators[MAX_ALLOCATORS];
static size_t num_allocators = 0;
void register_allocator(node_allocator *na) {
  // Check if already registered
  for (size_t i = 0; i < num_allocators; i++) {
    if (all_allocators[i] == na) {
      return; // Already registered
    }
  }

  if (num_allocators < MAX_ALLOCATORS) {
    all_allocators[num_allocators++] = na;
    // fprintf(stdout, "curr allocator count: %zu\n", num_allocators);
  } else {
    // fprintf(stderr, "Exceeded max allocators, curr: %zu\n", num_allocators);
    abort();
  }
}
void init_allocator(node_allocator *allocator, size_t capacity,
                    size_t node_size, bool in_cxl) {
  allocator->node_size = node_size;
  allocator->capacity = capacity;
  allocator->used = 0;

  size_t region_size = node_size * capacity;
  void *base = mmap(NULL, region_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  unsigned long node_mask = 1UL << (in_cxl ? CXL_MASK : LOCAL_MASK);
  long mbind_ret =
      mbind(base, region_size, MPOL_BIND, &node_mask, sizeof(node_mask) * 8, 0);
  if (mbind_ret != 0) {
    perror("mbind");
    exit(1);
  }
  allocator->base_addr = base;

  allocator->bitmap = calloc(capacity, sizeof(uint8_t));
  assert(allocator->bitmap);
  allocator->free_stack = malloc(capacity * sizeof(size_t));
  for (size_t i = 0; i < capacity; ++i) {
    allocator->free_stack[i] = capacity - 1 - i; // Fill stack in reverse
  }
  allocator->free_top = capacity;
  register_allocator(allocator);
  printf("allocator init for node_size: %zu, allocator addr: %p\n", node_size,
         allocator);
}

void *alloc_node_cus(node_allocator *allocator) {
  if (allocator->free_top == 0) {
    fprintf(stderr, "out of memory of allocator: %p\n", allocator);
    // return NULL;
    abort();
  }

  size_t index = allocator->free_stack[--allocator->free_top];
  allocator->bitmap[index] = 1;
  allocator->used++;

  void *ptr = (char *)allocator->base_addr + index * allocator->node_size;
  memset(ptr, 0, allocator->node_size); // Simulate calloc
  return ptr;
}

void free_node(node_allocator *allocator, void *node_ptr) {
  uintptr_t offset = (uintptr_t)node_ptr - (uintptr_t)allocator->base_addr;
  size_t index = offset / allocator->node_size;

  if (index >= allocator->capacity || allocator->bitmap[index] == 0) {
    fprintf(stderr, "Invalid free: %p\n", node_ptr);
    abort();
  }

  allocator->bitmap[index] = 0;
  allocator->used--;
  allocator->free_stack[allocator->free_top++] = index;
}

void debug_allocator(node_allocator *allocator) {
  printf("=== Allocator Debug ===\n");
  printf("Capacity     = %zu\n", allocator->capacity);
  printf("Node size    = %zu\n", allocator->node_size);
  printf("Used count   = %zu\n", allocator->used);
  printf("Free slots   = %zu\n", allocator->capacity - allocator->used);
  printf("Base address = %p\n", allocator->base_addr);
  printf("=======================\n");
}

void destroy_allocator(node_allocator *na) {
  if (!na)
    return;

  if (na->bitmap)
    free(na->bitmap);

  if (na->free_stack)
    free(na->free_stack);

  if (na->base_addr) {
    size_t total_size = na->node_size * na->capacity;
    if (munmap(na->base_addr, total_size) != 0) {
      perror("munmap failed in destroy_allocator");
    }
  }

  na->bitmap = NULL;
  na->free_stack = NULL;
  na->base_addr = NULL;
  na->node_size = 0;
  na->capacity = 0;
  na->used = 0;
}

#pragma GCC optimize("unroll-loops")
void free_node_auto(void *node_ptr) {
  uintptr_t addr = (uintptr_t)node_ptr;
  for (size_t i = 0; i < num_allocators; ++i) {
    node_allocator *na = all_allocators[i];
    uintptr_t base = (uintptr_t)na->base_addr;
    uintptr_t end = base + na->node_size * na->capacity;
    if (__builtin_expect(addr >= base && addr < end, 1)) {
      free_node(na, node_ptr);
      return;
    }
  }
  fprintf(stderr, "free_node_auto(): Invalid pointer %p\n", node_ptr);
  abort();
}