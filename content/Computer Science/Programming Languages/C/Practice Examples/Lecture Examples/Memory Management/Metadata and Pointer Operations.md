> [!ABSTRACT]
> 
> Memory management relies on Pointer Arithmetic to navigate the heap and Metadata (headers) to track block status. By using bitwise logic, we can store both the size of a block and its "Busy/Free" status in a single 8-byte word.

### 1. Pointer Arithmetic Review
In C, array notation is essentially "syntactic sugar" for pointer math. The compiler automatically scales the addition based on the size of the data type.

|**C Syntax**|**Pointer Equivalent**|**Description**|
|---|---|---|
|`a[i]`|`*(a + i)`|**Dereference**: Access the value at address `a` plus `i` offsets.|
|`&a[i]`|`a + i`|**Address**: Get the specific address of the $i$-th element.|
|`a[0]`|`*a`|**Base**: Access the very first element at the start of the pointer.|

**L-Value Definition**: An "L-Value" represents a memory location (an address) that can appear on the **left-hand side** of an assignment (`=`). If a pointer operation results in an L-Value, you can write data to that specific memory spot.

---
### 2. Malloc Metadata: The Header
Every block on the heap is preceded by an 8-byte **Header**. This header is the source of truth for the allocator.

![[Pasted image 20251107101504.png]]
#### The LSB (Least Significant Bit) Trick
Because our allocator rounds all block sizes to multiples of 8, the last 3 bits of any size are always `000`. We "borrow" the very last bit to act as a Boolean flag:
- **Odd Number (LSB = 1)**: The block is **Busy** (Allocated).
- **Even Number (LSB = 0)**: The block is **Free**.

> [!EXAMPLE]
> 
> - **Header = 33**: Binary `...00100001`. The size is 32 bytes ($33 \& \sim1$), and it is **Busy**.
>     
> - **Header = 108**: Binary `...01101100`. The size is 108 bytes, and it is **Free**.
>     

---
### 3. Allocation and Alignment

Computer architectures are optimized for aligned data. Our allocator ensures every request is rounded up to a multiple of 8, plus 8 bytes for the header itself.

**The "Rounding Up" Rule**: If a user requests 10 bytes:
1. Add 8 bytes for the header = 18.
2. Round up to the next multiple of 8 = **24 bytes total**.

```c
a = malloc(16); // Header 25 (16 user + 8 header + 1 flag)
b = malloc(40); // Header 49 (40 user + 8 header + 1 flag)
free(b);        // Header becomes 48 (Flag flipped to 0)
```

![[Pasted image 20251107105048.png]]

---
### 4. Logic of `free()`

When `free(ptr)` is called:
1. The allocator looks at `ptr - 1` (the word immediately preceding the data).
2. It performs a bitwise `AND` with a mask to flip the LSB to 0.
3. **Crucially**, the user data inside the block is **not wiped**. It remains there until another `malloc` overwrites it, which is why "Use After Free" bugs can sometimes appear to work correctly before failing later.

---
### Module Navigation
- **[[Implementation of Malloc]]**: See the code that implements this bitwise logic.
- **[[Coalescing Heap]]**: How the allocator uses these headers to merge adjacent free blocks.
- **[[Memory and References]]**: Reviewing how addresses are represented in hex.