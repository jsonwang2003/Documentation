> [!ABSTRACT]
> 
> At the lowest level, all data is represented as a series of bits (0s and 1s). While high-level C code works with integers and strings, systems programming often requires "Masking" and "Shifting" to extract or combine specific pieces of information within a single byte.
### 1. Bitwise Masking
A **Bitmask** is a value used to "filter" specific bits out of a number. By using the **Bitwise AND (`&`)** operator, we can force certain bits to zero while leaving others unchanged.

**Example: The `last4` Function**

```c
uint8_t last4(uint8_t n){
    // 0x0F or 0b00001111 is the mask
    return 0b00001111 & n;
}
```

- **Logic**: If a mask bit is `0`, the result bit is always `0`. If a mask bit is `1`, the result bit matches the input `n`.
- **Execution**:
    - `last4(0b10101100)` $\rightarrow$ `0b00001100` (Decimal 12)
    - `last4(0b10100000)` $\rightarrow$ `0b00000000` (Decimal 0)

---
### 2. Bitwise Shifting and Combining
To pack multiple pieces of data into one byte, we use **Bitwise OR (`|`)** and **Shifts (`<<` or `>>`)**.
- **Left Shift (`<<`)**: Moves bits to the left, filling the right side with zeros. Shifting left by 4 effectively moves the "last 4" bits into the "first 4" positions.
- **Bitwise OR (`|`)**: Combines two values. If either bit is `1`, the result is `1`.

**The combine Process:**
We want to take the last 4 bits of a and make them the first 4 bits of our result, then take the last 4 bits of b and make them the last 4 bits of our result.

```c
uint8_t combine(uint8_t a, uint8_t b){
    return (last4(a) << 4) | last4(b);
}
```

**Trace for `combine(0b10101111, 0b11000011)`:**
1. **Extract `a`**: `last4(0b10101111)` $\rightarrow$ `0b00001111`
2. **Shift `a`**: `0b00001111 << 4` $\rightarrow$ `0b11110000`
3. **Extract `b`**: `last4(0b11000011)` $\rightarrow$ `0b00000011`
4. **OR them**: `0b11110000 | 0b00000011` $\rightarrow$ `0b11110011`

---
### 3. Practical Applications
Bit manipulation is not just a mathematical exercise; it is used throughout this course to manage memory:
- **Malloc Headers**: As seen in [[Metadata and Pointer Operations]], we use the Bitwise AND (`&`) and Bitwise NOT (`~`) to flip the "Busy/Free" bit in a block header without changing the size value.
- **Permissions**: Unix file permissions (Read, Write, Execute) are stored as individual bits (e.g., `0b111` for `rwx`).
- **Hardware Flags**: Communicating with CPU registers often requires setting or clearing a single bit in a specific memory address.

---
### Common Bitwise Operators Reference

|**Operator**|**Name**|**Action**|
|---|---|---|
|`&`|**AND**|`1` if both bits are `1`. Used for masking.|
|`\|`|**OR**|`1` if either bit is `1`. Used for combining.|
|`^`|**XOR**|`1` if bits are different. Used for toggling.|
|`~`|**NOT**|Flips all bits (`0` becomes `1`, `1` becomes `0`).|
|`<<`|**Left Shift**|Moves bits left; multiplies by $2^n$.|
|`>>`|**Right Shift**|Moves bits right; divides by $2^n$.|
