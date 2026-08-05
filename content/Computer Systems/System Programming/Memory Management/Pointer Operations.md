> [!ABSTRACT]
> 
> A pointer is simply a variable that holds a memory address. To navigate through data, C uses Pointer Arithmetic, where adding an integer to a pointer moves it by a multiple of the size of the type it points to.

### 1. Address Calculation (Scaling)
When you index into a pointer or perform arithmetic, C automatically scales the offset.

Formula for Address:

$$\text{Address} = \text{base\_address} + (\text{sizeof(Type)} \times n)$$

#### Example Comparison: `int32_t` vs `uint8_t`

|**Pointer Type**|**Base Address**|**Operation**|**Size of Type**|**Resulting Address**|
|---|---|---|---|---|
|`int32_t* a`|`0x100`|`a + 3`|4 bytes|`0x10C` ($100 + 12$)|
|`uint8_t* b`|`0x200`|`b + 3`|1 byte|`0x203` ($200 + 3$)|

---
### 2. Array Syntax vs. Pointer Arithmetic
In C, the array subscript operator `[]` is actually **syntactic sugar** for pointer arithmetic and dereferencing.
- **The Equivalence**: `a[i]` is exactly the same as `*(a + i)`.
- **The Pointer**: A pointer does **not** represent an array; it only represents a starting address. It has no internal knowledge of the "size" or "length" of the data it points to.

|**Array Syntax**|**Pointer Arithmetic**|**Action**|
|---|---|---|
|`a[0]`|`*a`|Get the value at the base address.|
|`a[3]`|`*(a + 3)`|Get the value 3 elements away.|
|`&a[5]`|`a + 5`|Get the address 5 elements away.|

---
### 3. Why Use Pointer Arithmetic?
While array syntax is often more readable, pointer arithmetic is powerful when dealing with **offsets** and **sub-sections** of data.

> [!TIP]
> 
> Functional Advantage: If a standard library function (like strlen or strcpy) takes a starting pointer, you can pass a + 5 to make the function ignore the first 5 elements of your data without copying the array.

---
### 4. Exercise Review
1. **Setting `0x...114` via `int32_t* a`**:
    - Target is 20 bytes away from `0x...100` ($114 - 100 = 14_{hex} = 20_{dec}$).
    - Since each `int32_t` is 4 bytes, $20 / 4 = 5$.
    - Result: `a[5] = 37;` or `*(a + 5) = 37;`
2. **Setting `0x...214` via `uint8_t* b`**:
    - Target is 20 bytes away from `0x...200`.
    - Since each `uint8_t` is 1 byte, $20 / 1 = 20$.
    - Result: `b[20] = 37;` or `*(b + 20) = 37;`

---
### Module Navigation
- **[[Metadata and Pointer Operations]]**: How bitwise flags are hidden inside these addresses in `malloc`.
- **[[Memory and References]]**: Review how `&` and `sizeof` interact with these pointers.
- **[[Implementation of Malloc]]**: See how `p[-1]` uses negative pointer arithmetic to find a block header.