There are a lot of details that go into these implementations – we don't expect that anyone could internalize all the details from the documentation above quickly. However, there are a few key details that are really interesting to study. In this project, we'll explore how to implement a basic version of `malloc` and `free` that would be sufficient for many of the programs we have written.

Specifically, we will practice the following concepts in C:
- bitwise operations,
- pointer arithmetic,
- memory management, and of course,
- using the terminal and vim.

## Task Specification
### `vmalloc`

```c
void *vmalloc(size_t size);
```

The `size_t size` parameter specifies the number of bytes the caller wants. This has the same meaning as the usual `malloc` we have been using: the returned pointer is to the start of `size` bytes of memory that is now exclusively available to the caller.

Note the `void *` return type, which is the same as the usual `malloc`. `void *` is not associated with any concrete data type. It is used as a generic pointer, and can be implicitly assigned to any other type of pointer.

This is how `malloc` (and our `vmalloc`) allow assigning the result to any pointer type:

```c
int *p = malloc(sizeof(int) * 10);
char *c = malloc(sizeof(char) * 64);
```

If `size` is not greater than `0`, or if the allocator could not find a heap block that is large enough for the allocation, then `vmalloc` should return `NULL` to indicate no allocation was made.

Your implementation of `vmalloc` must:
- Use a “best fit” allocation policy
- Always return an address that is a multiple of **16**
- Always allocate space on **16-byte** boundaries
- Ensure that all blocks, either free or allocated, have correct block headers and footers

> [!NOTE]
> Best Fit Allocation Policy
> 
> Many allocation _policies_ are possible – choosing the first block that fits, choosing the last block that fits, choosing the most-recently-freed block that fits, and so on. The one you **must** implement for this PA is the best-fit policy.
> 
> That is, once you have determined the size requirement of the desired heap block, you need to traverse the entire heap to find the best-fitting block – the smallest block that is large enough to fit the allocation. If there are multiple candidates of the same size, then you should use the first one you find.
> 
> If the best-fitting block you find is larger than what we need, then we need to **split** that block into two. For example, if we are looking for a 16-byte block for `vmalloc(4)`, and the best fitting candidate is a 64-byte block, then we will split it into a 16-byte block and a 48-byte block. The former is allocated and returned to the caller, the latter remains free. Make sure to create a block header for any new blocks, and update block headers of modified blocks.

After updating all the relevant metadata, `vmalloc` should return a pointer to the start of the payload. Do not return the address of the block header!

### `vmfree`

```c
void vmfree(void* ptr)
```

The `vmfree` function expects a **16-byte** aligned address obtained by a previous call to `vmalloc`. If the address ptr is `NULL`, then vmfree does not perform any action and returns directly. If the block appears to be a free block, then `vmfree` does not perform any action and returns.

For allocated blocks, `vmfree` updates the block metadata to indicate that it is free, and _coalesces_ with adjacent (previous and next) blocks if they are also free.

#### Coalescing freed blocks
Just freeing a block is not necessarily enough, since this may leave us with many small free blocks that can't hold a large allocation. When freeing, we also want to check if the next / previous block in the heap are free, and coalesce with them to make one large free block.

If you are coalescing two blocks, remember to update both the header and the footer!

#### Block Footers / Previous Bit
You will need to update both your `vmalloc` and your `vmfree` implementation to add code for creating/updating accurate footers, and making sure the "previous block busy" bit is correct.

---
## Starter Code
#### The Library
- `vmlib.h`: This header file defines the public interface of our library. Other programs wishing to use our library will include this header file.
- `vm.h`: This header file defines the internal data structures, constants, and helper functions for our memory management system. It is not meant to be exposed to users of the library.
- `vminit.c`: This file implements functions that set up our “heap”.
- `vmalloc.c`: This file implements our own allocation function called `vmalloc`.
- `vmfree.c`: This file implements our own free function called `vmfree`.
- `utils.c`: This file implements helper functions and debug functions.
#### Testing
- `vmtest.c`: This file is not a part of the library. It defines a main function and uses the library functions we created. We can test our library by compiling this file into its own program to run tests. You can write code in the `main()` function here for testing purposes.
- `tests/`: This directory contains small programs and other files which you should use for testing your code. We will explain this in more detail in a later section.
### Compiling the Starter Code
To compile, run `make` in the terminal from the root directory of the repository. You should see the following items show up in your directory:

- `libvm.so`: This is a dynamically linked library, i.e., our own memory allocation system that can be linked to other programs (e.g., `vmtest`). The interface for this library is defined in `vmlib.h`. The `Makefile`s handle compiling with a `.so` for you; we set it up this way to match how production systems include libraries like `stdlib`
- `vmtest`: This executable is compiled from `vmtest.c` with libvm.so linked together. It uses your memory management library to allocate/free memory.
- The starter code in `vmtest.c` is very simple: it calls `vminit()` to initialize our simulated “heap”, and calls the `vminfo()` function to print out the contents of the heap in a neatly readable format. Run the `vmtest` executable to find out what the heap looks like right after it’s been initialized! (Hint: it’s just one giant free block.)
#### Reading the Starter Code
For this programming assignment, we want the addresses returned by `vmalloc` to always be 16-byte aligned (i.e., at an address ending in `0x0`). Since the metadata (block header) is 8 bytes in size, block headers will therefore always be placed at addresses ending in `0x8`. To maintain this alignment, the first block header (`*heapstart`) starts at an offset of 8, and all blocks going forward must have sizes that are multiples of 16 bytes.

Block sizes here refer to the size of the block **including** the header. The smallest possible block is 16 bytes, and would consist of an 8 byte header followed by 8 bytes of allocatable memory. When free, the last 8 bytes of the block would instead contain the block footer, to be used for coalescing free blocks.

Begin by reading through the `vm.h` file, where we define the internal data structures for the heap. This is where you will find the all-important block headers and block footers. Focus on understanding how struct `block_header` is used. You will see it in action later.

Next, open `vminit.c`. This is a very big file containing functions that create and set up the simulated heap for this assignment. Find the `init_heap()` function, and read through the entire thing to understand how it is setting up the heap. There is pointer arithmetic involved, you will need to do similar things for implementing `vmalloc` and `vmfree`, so make sure you have a solid understanding of that. (Why is it necessary to cast pointers to different types?)

Just like production allocators, we create a large chunk of memory as our heap using `mmap`, and allocations/frees are performed in that memory region.

Once you understand what the `init_heap()` function is doing, open up `utils.c`. Here we have implemented the function `vminfo()`, which will be your ally throughout this PA. This function traverses through the heap blocks and prints out the metadata in each block header. You should find inspiration for how to write your own `vmalloc` function here. Look at how it manipulates the pointer to jump between blocks!

---
## Testing
### Test Programs
In the repository, you will find the `tests/` directory, where we have provided some small testing programs that test your library function.

By coding in to the `tests/` directory and running make, you can compile all these test programs and run them yourself. These programs are what we will be using in the auto grader as well.

For example, here’s the code for the very first test: `alloc_basic.c`:
```c
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "vmlib.h"
#include "vm.h"

/*
 * Simplest vmalloc test.
 * Makes one allocation and confirms that it returns a valid pointer,
 * and the allocation takes place at the beginning of the heap.
 */
int main()
{
    vminit(1024);

    void *ptr = vmalloc(8);
    struct block_header *hdr = (struct block_header *)
                               ((char *)ptr - sizeof(struct block_header));

    // check that vmalloc succeeded.
    assert(ptr != NULL);
    // check pointer is aligned.
    assert((uint64_t)ptr % 16 == 0);
    // check position of malloc'd block.
    assert((char *)ptr - (char *)heapstart == sizeof(struct block_header));
    // check block header has the correct contents.
    assert(hdr->size_status == 19);

    vmdestroy();
    return 0;
}
```

In this test, we simply make one `vmalloc()` call, and we verify that the allocation returned a valid pointer at the correct location in the heap.

These tests use the `assert()` statement to check certain test conditions. If your code passes the tests, then nothing will be printed. If one of the assert statements fail, then you will see an error.
### Test Images
When you are writing vmalloc, you may wonder how you can test the allocation policy fully when you don’t have the ability to free blocks to create a more realistic allocation scenarios (i.e., a heap with many different allocated/free blocks to search through).

To help you with that, we have created some images in the starter code in the tests/img directory. In your test programs, instead of calling `vminit()`, you can call the `vmload()` function instead to load one of these images.

Our `alloc_basic2` program uses one such image. If you open `tests/alloc_basic2.c`, you will see that it creates the simulated heap using the following function call:

```c
vmload("img/many_free.img");
```

If you load this image and call `vminfo()`, you can see exactly how this image is laid out:

```txt
vmload: heap created at 0x7c2abd1da000 (4096 bytes).
vmload: heap initialization done.
---------------------------------------
 #      stat    offset   size     prev
---------------------------------------
 0      BUSY    8        48       BUSY
 1      BUSY    56       48       BUSY
 2      FREE    104      48       BUSY
 3      BUSY    152      32       FREE
 4      FREE    184      32       BUSY
 5      BUSY    216      48       FREE
 6      FREE    264      128      BUSY
 7      BUSY    392      112      FREE
 8      BUSY    504      32       BUSY
 9      FREE    536      112      BUSY
 10     BUSY    648      352      FREE
 11     BUSY    1000     304      BUSY
 12     BUSY    1304     336      BUSY
 13     FREE    1640     320      BUSY
 14     BUSY    1960     288      FREE
 15     BUSY    2248     448      BUSY
 16     BUSY    2696     256      BUSY
 17     BUSY    2952     96       BUSY
 18     BUSY    3048     368      BUSY
 19     FREE    3416     672      BUSY
 END    N/A     4088     N/A      N/A
---------------------------------------
Total: 4080 bytes, Free: 6, Busy: 14, Total: 20
```

You can use this image to test allocating in a more realistic heap.

Three images exist in total:
- `last_free.img`: the last block is free,
- `many_free.img`: many blocks are free,
- `no_free.img`: no block is free (use this to test allocation failure).

---
## Design Questions
_Heap fragmentation_ occurs when there are many available blocks that are small-sized and not adjacent (so they cannot be coalesced).
1. Give an example of a program (in C or pseudocode) that sets up a situation where a 20-byte `vmalloc` call **fails** to allocate despite the heap having (in total) over 200 bytes free across many blocks. Assume all the policies and layout of the allocator from the PA are used (best fit, alignment, coalescing rules, and so on)

```c
int main()
{
    vminit(4096); // comment this out if using vmload()

    void* blocks[256];

    for(int i = 0; i < 256; i++){
            blocks[i] = vmalloc(8);
    }

    for(int i = 0; i < 256; i += 2){
            vmfree(blocks[i]);
    }

    vminfo();

    void* ptr = vmalloc(20);
    if(!ptr){
            printf("vmalloc(20) fails since no free block can be merged and none have the size to allocate 20 bytes");
    } else {
            printf("somehow vmalloc(20) succeeds");
    }

    //vminfo(); // print out how the heap looks at this point for easy visualization

    vmdestroy(); // frees all memory allocated by vminit() or vmload()
    return 0;
}
```

2. Give an example of a program (in C or pseudocode) where all the allocations succeed if using the _best fit_ allocation policy (like in this PA), but some allocations would fail due to fragmentation if the _first fit_ allocation policy was used instead, keeping everything else the same.

```c
int main()
{
        vminit(4096);
        vminfo();

        // Create the blocks
        void* blocks[5];
        blocks[0] = vmalloc(16);
        blocks[1] = vmalloc(8);
        blocks[2] = vmalloc(3980);
        blocks[3] = vmalloc(8);
        blocks[4] = vmalloc(8);

        vminfo();

        // Free the first and last block
        vmfree(blocks[0]); // size = 32
        vmfree(blocks[4]); // size = 16

        vminfo();

        // Attempt to allocate more
        void* ptr1 = vmalloc(8); // size = 16
        void* ptr2 = vmalloc(16); // size = 32

        /*
        * With the best fit policy, ptr1 will be placed at the last block, and ptr2 will be placed at the first block
        * With the first fit policy, ptr1 will be placed at the first block, and ptr2 will not be placed since now the heap has 2 separate 16-byte blocks, which are not enough to place a 32-byte block.
        */


        vminfo();

        vmdestroy();
        return 0;
}
```

---
## Code
Repository found [here](https://github.com/ucsd-cse29-fa25/pa4-jsonwang2003)
### `vmalloc.c`
```c
#include "vm.h"
#include "vmlib.h"

// Given a void* pointer, return the same pointer if it is 16-byte aligned, otherwise return the next higher 16-byte aligned address.
void* align_me16(void* ptr){
	uint64_t a = (uint64_t) ptr;
	int dif = a % 16;
	if (dif == 0) {return ptr;}
	return (ptr + 16 - dif);
}


int is_aligned16(void* ptr){
	uint64_t a = (uint64_t) ptr;
	return (a % 16 == 0) ? 1 : 0;
}

// Given a size_t number of bytes, return the minimum block size needed to account for the header, the allocation request, and 16-byte alignment.
size_t min_block_size(size_t bytes){
	if (bytes == 0) return 0;
    int result = 8;
    result += (bytes % 8 == 0) ? bytes : bytes + 8 - bytes % 8;
    if (result % 16 == 0) return result;
    return (result + 8);
}

// Return the address to the header of a free "best fit" block of at least size bytes
void *find_min_block(size_t size){
	struct block_header *block = heapstart;
	struct block_header* result = NULL;

	while(block->size_status != VM_ENDMARK){
		size_t header = block->size_status;
		if(header >= size && (header & VM_BUSY) == 0){
			if(result == NULL || result->size_status > header){
				result = block;
			}
		}
		block = (struct block_header*)((char*)block + BLKSZ(block));
	}

	return result;
}

// Split the found free block to one that is busy and another that is free
void split_block(uint64_t* block_start, size_t newsize){
	uint64_t header = block_start[0] & ~VM_PREVBUSY;
	if(header == newsize){
		block_start[0] |= 1;
		if(block_start[newsize / 8] != VM_ENDMARK){
			block_start[newsize / 8] |= VM_PREVBUSY;
		}
		return;
	}
	size_t free_size = header - newsize;
	block_start[0] = newsize + (block_start[0] & VM_PREVBUSY) + VM_BUSY;
	if(block_start[newsize / 8] != 1){
		block_start[newsize / 8] = free_size + VM_PREVBUSY;
		
		// updating the footer in freed block
		int footer_index = (newsize + free_size - 8) / 8;
		block_start[footer_index] = free_size;
	}
}

// Align user_request size in multiplies of 8 (including header)
size_t block_size_of(size_t request_size){
	return ROUND_UP((request_size + 8), BLKALIGN);
}

// malloc a size in the heap memory for user
void *vmalloc(size_t size)
{
	// Check for undefined input
	if (size <= 0){
		return NULL;
	}

	// Find the block in heap that can is the "best fit" for the requested size
    size_t block_size = block_size_of(size);
	uint64_t* block_to_use = find_min_block(block_size);

	// Cannot find a fit for the requested size
	if (block_to_use == NULL){
		return NULL;
	}

	// Split the block to that of being free and that of being busy
	split_block(block_to_use, block_size);

	// return the start of the payload space
	uint64_t* user_block = &block_to_use[1];
	return user_block;
}
```
### `vmfree`
```c
#include "vm.h"
#include "vmlib.h"

#define SIZE_MASK (~0xF)
// Get block size from footer
size_t get_prev_block_size(struct block_header *block){
	struct block_footer* footer = (struct block_footer*) block;
	footer -= 1;
	return footer->size;
}

// Write Block Footer
void write_block_footer(struct block_header* block){
	size_t size = block->size_status & SIZE_MASK;
	struct block_footer* footer = (struct block_footer*)((char*)block + size - sizeof(struct block_footer));
	footer->size = size;
}

// get next block's address
struct block_header* next_block_addr(struct block_header* start){
	uint64_t size = start->size_status & ~0x3;
	return (struct block_header*)((char*)start + size);
}

// Given two adjacent block_header*, combine them so it becomes 1 larger block
void coalesce_blocks(struct block_header* start, struct block_header* end){
	size_t size = (start->size_status & ~3) + (end->size_status & ~3);
	start->size_status = size + (start->size_status & VM_PREVBUSY);
	write_block_footer(start);
}

/**
 * The vmfree() function frees the memory space pointed to by ptr,
 * which must have been returned by a previous call to vmmalloc().
 * Otherwise, or if free(ptr) has already been called before,
 * undefined behavior occurs.
 * If ptr is NULL, no operation is performed.
 */
void vmfree(void *ptr)
{
	if(ptr == NULL){
		return;
	}
	struct block_header* current = (struct block_header*)((uint64_t*)ptr - 1);
	if((current->size_status & VM_BUSY) == 0){
		return;
	}

	// Free the current block
	current->size_status &= ~VM_BUSY;
	struct block_header* next = next_block_addr(current);
	next->size_status &= ~VM_PREVBUSY; // update next block 
	write_block_footer(current); // write to footer

	// Check if the next block is free
	if((next->size_status & VM_BUSY) == 0){
		coalesce_blocks(current, next);
	}

	// Check if previous block is free
	if((current->size_status & VM_PREVBUSY) == 0){
		size_t prev_size = get_prev_block_size(current);
		struct block_header* previous = (struct block_header*)((char*)current - prev_size);
		coalesce_blocks(previous, current);
	}
}
```
