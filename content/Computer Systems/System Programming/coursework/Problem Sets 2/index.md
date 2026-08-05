---
title: Problem Sets 2
---
> [!ABSTRACT]
> 
> This problem set focuses on Stream Processing and Command Line Interaction. It introduces the mechanics of reading from standard input, parsing command-line arguments, and implementing data integrity checks using the SHA256 hashing algorithm.

---
## Problem Categories
### Standard Input and Output (I/O)
- **[[Count Lines from Standard Input]]**: Implementing a utility to count newline characters in a byte stream.
- **[[Count Occurrences from Standard Input]]**: Searching for and tallying specific patterns or characters in redirected input.
- **[[SHA256 Standard Input]]**: Reading a continuous stream of data to generate a cryptographic hash.
- **[[Count 0 Bits]]**: Implementing population count logic to tally set bits in a word.
### Command Line Arguments
- **[[Run is_ascii on Command Line Argument]]**: Integrating bitwise validation with the `argc` and `argv` interface.
- **[[SHA256 Argument]]**: Passing specific strings via the terminal to be processed by the hashing engine.
- **[[String Capitalization Match]]**: Comparing and transforming strings passed as program arguments.
### Data Transformation and Hexadecimal
- **[[Convert Hex]]**: Translating human-readable hexadecimal strings into numeric values.
- **[[Hexadecimal to Bytes Converter]]**: Mapping pairs of hex characters to their corresponding raw byte representations.
- **[[SHA256 Hash Program (Hex)]]**: Outputting cryptographic digests in standard hexadecimal format.
- **[[Parse Escapes]]**: Handling special character sequences (like `\n` or `\t`) within string inputs.
### String Logic and Filtering
- **[[Is Palindrome]]**: Implementing two-pointer logic to verify symmetrical string properties.
- **[[Reverse String]]**: Managing in-place memory swaps to flip the order of characters.
- **[[Strip Line Comments]]**: Parsing a file or stream to identify and remove content following specific markers (e.g., `//`).
- **[[String with Exclamation Point]]**: Simple string manipulation and concatenation exercises.
- **[[Capitalization Variants]]**: Exploring different casing rules and their bitwise implementations.
### Hashing Control Flow
- **[[SHA256 Hash Checker]]**: Comparing a generated hash against a "known good" value to verify file integrity.
- **[[SHA256 Loop]]**: Iteratively hashing data, a fundamental concept in proof-of-work or repeated encryption.

---
## Technical Summary

|**Concept**|**C Implementation**|**Usage**|
|---|---|---|
|**Stream Reading**|`getchar()` or `fgets()`|Processing data byte-by-byte or line-by-line.|
|**CLI Arguments**|`int main(int argc, char *argv[])`|Receiving user input directly from the shell.|
|**Hashing**|`sha256_update()`|Accumulating data into a fixed-size cryptographic digest.|

---
## Related Toolkits

- **[[Computer Science Introduction/Programming Languages/C/Syntax/index|C Syntax]]**: Deepens understanding of arrays (`char *`) and loop structures for stream processing.
- **[[Computer Science Theory/Discrete Structures/Counting/index|Counting]]**: Relevant for understanding the collision resistance and output space ($2^{256}$) of SHA256.
- **[[Computer Systems/System Programming/Intro to Systems Programming/index|Lecture: Memory and System Calls]]**: Background on how the OS passes arguments and handles file descriptors.