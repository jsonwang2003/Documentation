In C, the size of a `struct` isn't always a simple sum of its parts. While the math seems straightforward, the compiler often adds invisible "filler" bytes to ensure that the CPU can access data as quickly as possible. This section explains the difference between **Array Members** and **Pointer Members**, as well as the concept of **Padding**.

---
## 1. Array vs. Pointer Members
The way you define a member drastically changes the memory footprint of the struct itself.
### Array Members (Fixed Size)
When you use an array like `char contents[100]`, the space for those 100 characters is **in-lined** directly into the struct.
- **Pros**: All data is in one contiguous block; no extra allocation needed.
- **Cons**: The struct size is rigid and cannot grow.
### Pointer Members (Dynamic Size)
When you use a pointer like `char* contents`, the struct only stores the **memory address** (usually 8 bytes on modern systems).
- **Pros**: The struct stays small; the actual string can be any length (allocated on the Heap).
- **Cons**: Requires manual memory management (`malloc` and `free`).

---
## 2. Why is `sizeof(String2)` sometimes 16?

You noted that `sizeof(String2)` could be **12** or **16**. In systems programming, we call the difference **Padding**.
### Data Alignment
CPUs are designed to read data most efficiently when it starts at an address that is a multiple of its own size. This is called **Alignment**.
- An `int` (4 bytes) should start at an address divisible by 4.
- A `char*` (8 bytes) should start at an address divisible by 8.
### The Padding Logic
If we look at `struct String2`:
1. **`int len`**: Takes up bytes 0–3.
2. **The Gap**: If the pointer `contents` started at byte 4, it wouldn't be "8-byte aligned."
3. **Padding**: The compiler inserts 4 "empty" bytes (bytes 4–7).
4. **`char* contents`**: Now starts at byte 8 and goes to byte 15.

Total size = **16 bytes**.

---
## 3. Calculation Summary

|**Struct**|**Member 1**|**Member 2**|**Raw Sum**|**Padded Size**|
|---|---|---|---|---|
|**Point**|`int` (4)|`int` (4)|8|**8**|
|**String**|`int` (4)|`char[100]` (100)|104|**104**|
|**String2**|`int` (4)|`char*` (8)|12|**16**|

> [!TIP]
> 
> To minimize padding and save memory, a common rule of thumb is to order your struct members from largest to smallest.

---
### Module Navigation
- **[[Structs in C]]**: Syntax and basic usage.
- **[[Dynamic Struct]]**: How to use `String2` with `malloc`.
- **[[Sizeof]]**: Reviewing how types are measured in 64-bit systems.