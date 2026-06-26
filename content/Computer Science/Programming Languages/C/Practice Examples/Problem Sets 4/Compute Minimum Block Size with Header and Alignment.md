## Task
Implement the function `min_block_size`
- Write a function that takes a `size_t` representing the number of bytes requested for allocation
- Each block must have a header of type `size_t` at the beginning of the block
- The total block size (header + requested bytes) must be rounded up to the next multiple of 16 bytes to ensure 16 byte alignment
### Function Signature
```c
// Given a size_t number of bytes, return the minimum block size needed to account for the header, the allocation request, and 16-byte alignment.
size_t min_block_size(size_t bytes);
```

---
## Example
| Requested Bytes | Minimum Block Size |
| --------------- | ------------------ |
| 0               | 0                  |
| 1               | 16                 |
| 8               | 16                 |
| 16              | 32                 |
| 24              | 32                 |
| 100             | 112                |
| 4096            | 4112               |

---
## Code
```c
size_t min_block_size(size_t bytes) {
    if (bytes == 0) return 0;
    int result = 8;
    result += (bytes % 8 == 0) ? bytes : bytes + 8 - bytes % 8;
    if (result % 16 == 0) return result;
    return (result + 8);
}
```