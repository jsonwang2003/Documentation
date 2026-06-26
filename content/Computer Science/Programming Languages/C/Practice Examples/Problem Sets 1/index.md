---
title: Problem Sets 1
---
> [!ABSTRACT]
> 
> This problem set serves as the introduction to Low-Level Data Representation. It covers the fundamental mechanics of bit manipulation, integer signedness, and the serialization of characters into the UTF-8 encoding standard.

---
## Problem Categories
### Bitwise Fundamentals
- **[[Bitwise Is Even]]**: Determining parity using bitwise AND instead of arithmetic modulo.
- **[[Count 1 Bits]]**: Implementing population count logic to tally set bits in a word.
- **[[Check if Bits Start with '10']]**: Pattern matching for specific bit sequences, essential for protocol headers.
- **[[Extract Least Significant 16 Bits]]**: Isolating the lower half of a 32-bit integer.
- **[[Extract Most Significant 16 Bits]]**: Isolating the upper half of a 32-bit integer.
- **[[Extract Lowest N Bits]]**: A generalized approach to creating bitmasks for variable-width data extraction.
### Integer Representations
- **[[Binary String to Signed 8-bit Integer]]**: Interpreting small-scale Two's Complement values.
- **[[Convert Binary String to Signed 32-bit Integer]]**: Handling standard integer signed-ness in C.
- **[[Convert Binary String to Unsigned 32-bit Integer]]**: Converting raw binary strings to positive integer magnitudes.
- **[[Printing Char and Integer Representations]]**: Exploring how the same 8-bit pattern is interpreted as a character vs. a numeric value.
- **[[Printing Char and Integer Representations Backwards]]**: Advanced iteration and bit-field inspection.
### Character Encodings (ASCII & UTF-8)
- **[[Is ASCII]]**: Validating the 7-bit standard within an 8-bit byte.
- **[[Capitalize ASCII]]**: Using bit-toggling to transform character case without branching.
- **[[To Lowercase]]**: Applying bitwise offsets to normalize character casing.
- **[[UTF-8 Codepoint Size]]**: Analyzing leading bytes to determine the length of a multi-byte character.
- **[[Encode UTF-8]]**: Mapping a Unicode codepoint into its variable-width byte sequence.
- **[[Count UTF-8 String Length]]**: Distinguishing between byte count and logical character count.
- **[[Find UTF-8 Codepoint at Index]]**: Navigating a variable-width stream to locate a specific character.
- **[[Extract a UTF-8 Substring by Codepoint Indices]]**: Implementing logical slicing on encoded data.
- **[[Check if a Codepoint is an Animal Emoji]]**: Validating specific hexadecimal ranges within the Unicode space.

---
## Technical Summary

|**Operation**|**C Syntax Example**|**Purpose**|
|---|---|---|
|**Masking**|`x & 0xFFFF`|Clear bits outside target range.|
|**Shifting**|`x >> 16`|Move high bits into low positions.|
|**Encoding**|`(cp >> 6)|0x80`|

---
## Related Toolkits
- **[[Syntax/index|C Syntax]]**: Essential for using bitwise operators (`&`, `|`, `~`, `^`, `<<`, `>>`).
- **[[Discrete Structures/Counting/index|Counting]]**: Used to determine the number of possible values in different bit-widths (e.g., $2^{16}$ vs $2^{32}$).
- **[[Practice Examples/Lecture Examples/Strings and Data/index|Lecture: Strings and Data]]**: Theoretical background on how these encodings were designed.