> [!ABSTRACT]
> 
> While standard English characters fit into 7 bits (ASCII), many international characters (like ê) require multiple bytes. UTF-8 is a variable-width encoding that uses the most significant bits of a byte to signal how many total bytes represent a single "symbol."

### 1. The Multi-Byte Nature of UTF-8
In your `inspect` output, the string `"Kiên"` has a logical length of 4 characters, but `strlen` reports a length of **5**.
- **K, i, n**: These are standard ASCII characters (starting with `0` in binary). They take **1 byte** each.
- **ê**: This is a multi-byte character. In binary, it is represented by two bytes: `11000011` and `10101010`.

---
### 2. Decoding the Byte Headers
UTF-8 uses a specific bit pattern at the start of a byte to tell the computer how to read the sequence. The function `get_size_of_symbol` implements this logic using **Bitmasks**:

|**Prefix Bits**|**Mask Used**|**Symbol Size**|**Description**|
|---|---|---|---|
|`0xxxxxxx`|`0b10000000`|**1 Byte**|Standard ASCII (0–127)|
|`110xxxxx`|`0b11100000`|**2 Bytes**|Multi-byte start (e.g., `ê`, `ñ`)|
|`1110xxxx`|`0b11110000`|**3 Bytes**|Most common Asian characters, symbols|
|`11110xxx`|`0b11111000`|**4 Bytes**|Emojis and rare mathematical symbols|
|`10xxxxxx`|`0b11000000`|**Continuation**|These bytes "belong" to the previous header|

> [!IMPORTANT]
> 
> Because of this, str[index] no longer necessarily refers to the "$i$-th character" of a string, but rather the "$i$-th byte." To find the next character, you must jump forward by the number of bytes returned by get_size_of_symbol.

---
### 3. Bitwise Analysis of `get_size_of_symbol`
The code uses bitwise AND to isolate the "header" bits of the character.

```c
if(c & 0b11100000 == 0b11000000) {return 2;}
```

- **The Mask (`0b11100000`)**: This ignores the last 5 bits of the character.
- **The Comparison**: If the first three bits are exactly `110`, we know we are at the start of a 2-byte sequence.

---
### 4. Implementation Trace: `"Kiên"`
Let's look at the bytes stored for "Kiên" at the index where `ê` starts (index 2):
1. **Index 2**: `11000011` (195)
    - `11000011 & 0b11100000` results in `11000000`.
    - Matches the 2-byte rule. **Size = 2**.
2. **Index 3**: `10101010` (170)
    - This is a **continuation byte** (starts with `10`). It does not count as a new symbol; it provides more data for the `ê`.
3. **Index 4**: `01101110` (110)
    - This is `n`. Starts with `0`. **Size = 1**.

---
### 5. Vocabulary & Formats
- **Code Point**: The unique number assigned to a character by the Unicode standard (e.g., U+00EA for `ê`).
- **Encoding (UTF-8)**: The specific way that number is turned into binary bytes.
- **`%hhb`**: A newer C format specifier (C23) to print integers in binary. If your compiler is older, you may need a custom function to see binary output.
