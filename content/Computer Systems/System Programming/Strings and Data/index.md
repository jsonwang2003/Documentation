---
title: "Strings and Data" 
description: "Index for Strings and Data, covering bitwise manipulation, signed vs unsigned numbers, bit masks, memory string concatenation, and Unicode/UTF-8 codepoints." 
aliases: 
- Strings and Data Hub
- Bitwise & UTF-8 
tags: 
- index 
- system-programming 
- bitwise 
- utf8
- unicode
---
> [!ABSTRACT]
> 
> This chapter bridge the gap between abstract data types and their binary reality. We explore how C manages text through memory addresses, the pitfalls of signed data types, and the bitwise logic required to decode modern international text standards like UTF-8.

### 1. Bit-Level Foundations
Before handling strings, we must master the manipulation of individual bits.
- **[[Bitwise Operators Review]]**: Mastering AND (`&`), OR (`|`), XOR (`^`), and Bit-Shifting (`<<`, `>>`) to isolate or combine data.
- **[[Signed vs Unsigned Numbers]]**: Understanding **Two's Complement** and how the most significant bit changes the numerical interpretation of a byte.
- **[[Signedness and Bit Masks]]**: Why `char` can be dangerous in comparisons and how to use `unsigned char` and masks to ensure logic remains consistent.

---
### 2. Memory and Strings
How C manages sequences of characters on the Stack.
- **[[String Concatenation and Memory]]**: Managing the **Null Terminator (`\0`)**, calculating buffer sizes, and why returning local arrays leads to "Dangling Pointers."
- **[[String Concatenation and Memory#1. The Out Parameter Pattern|Out Parameter]]**: Learning the pattern of passing a destination buffer into a function rather than returning a new one.

---
### 3. Internationalization (Unicode)
Moving beyond the 128-character limit of ASCII into the global standard of UTF-8.
- **[[Unicode and UTF-8 Encoding]]**: Understanding variable-width encoding. How a single "symbol" can span 1 to 4 bytes depending on its header bits.
- **[[UTF-8 Codepoint]]**: The math of decoding. Stripping structural header bits and shifting payload bits to reconstruct a character's unique Unicode identity.
- [[Computer Systems/System Programming/Strings and Data/UTF-8|UTF-8]]: A quick notesheet of basic UTF-8 codes

---
### 4. Practical Application: String Inspection
Tools and techniques for seeing what is actually inside a string variable.
- **Format Specifiers**: Using `%p` for addresses, `%hhb` for binary, and `%hhu` for unsigned byte values.
- **Decoding Logic**: implementing `get_size_of_symbol` and `codepoint_of` to navigate multi-byte strings correctly.

---
### Quick Reference: UTF-8 Header Patterns

|**Leading Byte Bits**|**Bytes in Symbol**|**Data Bits (Payload)**|
|---|---|---|
|`0xxxxxxx`|1|7 bits|
|`110xxxxx`|2|5 bits (+ 6 from next)|
|`1110xxxx`|3|4 bits (+ 12 from next)|
|`11110xxx`|4|3 bits (+ 18 from next)|


