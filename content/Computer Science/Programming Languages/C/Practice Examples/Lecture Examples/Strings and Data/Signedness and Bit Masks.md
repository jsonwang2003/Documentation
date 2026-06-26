> [!ABSTRACT]
> 
> In C, the char type is often signed by default (depending on the compiler and architecture). This means the most significant bit (MSB) acts as a sign bit, leading to unexpected results when performing "greater than" or "less than" comparisons on raw byte values.

---
### 1. The Signed Interpretation Error
When `char` is signed, it uses **Two's Complement** representation. A byte like `0xFF` is not interpreted as $255$, but as $-1$.
**The Comparison Breakdown:**
- **Logic**: We want to check if a character is in the standard ASCII range ($0$–$127$).
- **`c <= 128`**: If `c` is `0xFF` (signed $-1$), the expression `-1 <= 128` evaluates to **True**.
- **`c & 0b10000000`**: This checks if the 8th bit is set. If `0xFF` is masked, the result is `0b10000000`, which is **not 0**. This correctly identifies that the value is outside the standard 7-bit ASCII range.

---
### 2. The Truth Table Analysis
The discrepancy arises because the Bitwise AND looks at the raw bits, while the "Less Than" operator looks at the numerical value interpreted by the type.

|**Input (Hex)**|**Binary**|**Signed Value**|**c < 128**|**(c & 0x80) == 0**|
|---|---|---|---|---|
|**0x48**|`01001000`|72|True (1)|True (1)|
|**0xFF**|`11111111`|-1|**True (1)**|**False (0)**|
|**0x0A**|`00001010`|10|True (1)|True (1)|

> [!IMPORTANT]
> 
> Because -1 is less than 128, the "Less Than" check fails to catch non-ASCII characters if the input is treated as a signed integer.

---
### 3. The Fix: Explicit Casting
To ensure the comparison treats the byte as a value from $0$–$255$, you must cast the variable to an `unsigned char`.

```c
int8_t is_ascii_lessthan(char c){
    // Casting to unsigned char forces the interpretation of 0xFF as 255
    return ((unsigned char) c) < 128;
}
```

By casting to `unsigned`, `0xFF` becomes $255$. The expression `255 < 128` correctly evaluates to **False (0)**, aligning the behavior of the numerical check with the bitwise check.

---
### 4. Summary of Bitwise vs. Numerical Checks
- **Bitwise AND (`&`)**: Generally safer for checking specific bit patterns (like the "High Bit" in ASCII) because it ignores signedness.
- **Numerical Comparisons (`<`, `>`)**: Faster and more readable, but require careful attention to whether the type is `signed` or `unsigned`.