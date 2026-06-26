---
title: Problem Sets 4
---
> [!ABSTRACT] 
> This problem set explores the internal mechanics of **Dynamic Memory Allocation**. It focuses on the low-level implementation of `malloc` and `free`, specifically how to manage a heap through block headers, footers, pointer arithmetic, and alignment.

---
## The Heap Architecture
In this simulated environment, we use a specific set of rules to manage memory blocks:
- **Headers & Footers**: Every block has an 8-byte **Header**. Only free blocks have an 8-byte **Footer** to allow for backward traversal (coalescing).
- **Alignment**: All blocks are **16-byte aligned**. The smallest possible allocation is 16 bytes.
- **Size/Status Encoding**: The `size_t size_status` field uses the **Least Significant Bit (LSB)** to indicate if a block is busy (1) or free (0).
- **Endmark**: A sentinel block with `size_status = 0x1` (size 0, status busy) signals the end of the heap.

---
## How is heap structured for this Problem Set?
The simulated heap that we have created is a large chunk of memory that we can allocate or free. Here are some relevant blocks:
### Block Headers (`struct block_header`)
- Both allocated blocks and free blocks have block headers
- Block headers are 8 bytes
- The block header stores the size and the status of the block in a `size_t size_status` variable
	- The **least significant bit** represents the state of the current block
		- 0 → free
		- 1 → busy / allocated
	- The rest of the bits are used to store the size of the block
		- For allocated blocks → size = block header + payload allocated + padding to make block 16-byte aligned (round up)
		- For free blocks → size = block header + free space + block footer
### Block Footers (`struct block_footer`)
- Only free blocks have block footers
- Block footers are 8 bytes
- Block footers only stores the size, and not the status of the block
	- Since a block's header contains its size, we know how far forward to move to get to next block's header.
		- However, when coalescing backwards, we need to know the size of the previous block to get to its header
		- To avoid walking the entire list of blocks from the beginning, we write the block size to a footer in the last 8 bytes of the block
	- Since footers are only needed during coalescing, we only need to add footers to free blocks
		- This means that footers don't take up extra space in allocated blocks, and free blocks aren't using that space anyway
### Endmark
- To indicate that you have reached the end of the heap, there is special block at the end of the heap: **The Endmark**
- This block has a size of 0 but is indicated to be busy (Least significant bit is 1)
	- This block has a `size_status` of 0x1
### Diagram
Here is a diagram for heap

![[Pasted image 20251110111950.png]]

- `vmalloc` is a function you will need to implement in PA4.
## How do I access this heap?
Created a few fake heaps behind the scenes and gave them to you in the form of the `fake_heaps.o` object file. The exact structure of each heap is included in the diagrams of the `fake_heaps.h` header file
- You can traverse through the heap using the size stored in the `block_headers` and pointer arithmetic

> [!HINT]
> When doing pointer arithmetic, cast your pointers to `(char *)` first to avoid scaling by the size of struct

---
## Problem Categories
### Memory Alignment
- **[[Align Address to 16-Byte Boundary]]**: Logic for rounding an address up to the nearest 16-byte multiple.
- **[[Check if Address is 16-Byte Aligned]]**: Verifying that an address is a multiple of 16 (often using the bitwise check `addr % 16 == 0`).
### Header & Footer Metadata
- **[[Extract Block Size from Encoded Block Header]]**: Using bitwise masks to isolate the size bits from the status bit.
- **[[Extract Allocation Status from Encoded Block Header]]**: Checking the LSB to determine if a block is available.
- **[[Extract Previous Block's Allocation Status from Encoded Block Header]]**: Accessing metadata about the preceding neighbor.
- **[[Set Current Block Allocation Status]]**: Using bitwise OR/AND to toggle the status bit without changing the size.
- **[[Set Previous Block Allocation Status]]**: Updating neighbor metadata during allocation or freeing.
- **[[Write Block Footer]]**: Mirroring the header size into the last 8 bytes of a free block.
- **[[Get Previous Block Size from Footer]]**: Jumping backward in memory by reading the preceding block's footer.
### Block Traversal & Manipulation
- **[[Next Block Address]]**: Calculating the start of the next block by adding the current block's size to its pointer.
- **[[Split Block]]**: Dividing a large free block into an allocated chunk and a remaining free chunk.
- **[[Coalesce Blocks]]**: Merging adjacent free blocks to combat external fragmentation.
- **[[Compute Minimum Block Size with Header and Alignment]]**: Calculating the total footprint required for a specific payload.
### Heap Analysis & Pointers
- **[[Count Free Blocks]]**: Iterating through the implicit list from the start of the heap to the Endmark.
- **[[Number of Bytes Allocated]]** / **[[Number of Bytes Free]]**: Tallying the usage statistics of the current heap state.
- **[[Double Pointers]]**: Managing pointers to block pointers, essential for updating free list heads or managing complex structures.
- **[[Strings & Pointers]]**: Handling character data within the context of heap-allocated memory.
- **[[File Structure and Debugging]]**: Understanding how the `fake_heap` is modeled and how to use tools to inspect it.

---
## Technical Summary

|**Rule**|**Value**|**Note**|
|---|---|---|
|**Header Size**|8 Bytes|Contains `size_status`.|
|**Alignment**|16 Bytes|`(size + 7) & ~7` is a common alignment trick.|
|**LSB = 1**|Busy|Block is currently allocated.|
|**LSB = 0**|Free|Block is available for `malloc`.|

---
## Related Toolkits
- **[[Syntax/index|C Syntax]]**: Specifically **Pointer Arithmetic**. Remember to cast to `(char *)` to move byte-by-byte.
- **[[Discrete Structures/Counting/index|Counting]]**: Used for calculating offsets and total heap capacity.
- **[[Practice Examples/Lecture Examples/Memory Management/index|Lecture: Memory Management]]**: The theory of heap vs. stack and the design of `malloc`.