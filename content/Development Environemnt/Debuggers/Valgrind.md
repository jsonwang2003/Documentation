> [!INFO] Source Attribution
> These notes are based on the tutorial **[Using Valgrind to Find Memory Leaks](https://www.cprogramming.com/debugging/valgrind.html)** by Cprogramming.com.
## 1. Introduction & Installation

Valgrind runs your program in a monitored environment, tracking every call to memory allocation (`malloc`, `new`) and deallocation (`free`, `delete`).
### Quick Installation (Source)
1. **Decompress:** `bzip2 -d valgrind-XYZ.tar.bz2`
2. **Extract:** `tar -xf valgrind-XYZ.tar`
3. Build & Install: 
```bash
cd valgrind-XYZ
./configure
make
make install
```

---
## 2. Detecting Memory Leaks
Memory leaks are silent bugs that occur when allocated memory is never freed. Over time, these can cause a program to crash once system memory is exhausted.
### Basic Usage

To check for leaks, use the `memcheck` tool.

```Bash
valgrind --tool=memcheck ./program_name
```

A balanced program will show equal numbers of `allocs` and `frees`. If they differ, you have a leak.
### Detailed Analysis
To see exactly where the leak occurred, compile your code with the `-g` flag (to include debugging symbols) and run:

```Bash
valgrind --tool=memcheck --leak-check=yes ./program_name
```

> [!Tip] Pro Tip: 
> Use `--show-reachable=yes` to find absolutely every unpaired call to `new` or `malloc`.

---
## 3. Finding Invalid Pointer Usage
Valgrind detects when your code accesses memory addresses that it does not "own," such as writing past the end of an array.
- **Invalid Write:** Occurs when you try to change data outside allocated bounds.
- **Invalid Read:** Occurs when you try to access data outside allocated bounds.

> [!Example]- Example Error:
> If you `malloc(10)` but attempt to write to `x[10]`, Valgrind will trigger an "Invalid write of size 1" and provide a stack trace to the exact line of code.

---
## 4. Detecting Uninitialized Variables

Using a variable before assigning it a value can lead to unpredictable program behavior. Valgrind flags "Conditional jump or move depends on uninitialized value(s)."
- **Logic Tracking:** Valgrind is smart enough to track an uninitialized value even if it is passed through several functions (e.g., from `main` into a helper function).
- **Requirement:** You must test the specific branch of code containing the conditional statement for Valgrind to see the error.

---
## 5. Other Memory Errors
- **Double Frees:** Calling `free()` or `delete` twice on the same pointer.
- **Mismatched Deallocators:** Using the wrong method to free memory.
    - `malloc` must be paired with `free`.
    - `new` must be paired with `delete`.
    - `new[]` must be paired with `delete[]`.

---
## 6. Limitations & Caveats

|**Feature**|**Supported?**|**Note**|
|---|---|---|
|**Heap Memory**|Yes|Monitors `malloc`, `new`, etc.|
|**Stack Memory**|**No**|Does **not** check bounds for static arrays like `char x[10]`.|
|**Performance**|Variable|Can slow down code significantly and use 2x more memory.|
|**Edge Cases**|Limited|Will not detect buffer overflows from long input strings unless that specific memory is touched.|

> [!TIP]
> 
> If you suspect a bug in a static (stack) array, temporarily convert it to a dynamic (heap) array. Valgrind will then be able to perform bounds checking on it.

---
## Valgrind Cheat Sheet
![[Valgrind_Quick_Reference_Guide.docx.pdf]]

## Summary
Valgrind is an essential tool for C/C++ developers on Linux. By running your executable in Valgrind's environment, you can proactively catch **unpaired memory calls**, **invalid pointer math**, and **uninitialized variables** before they become difficult-to-trace production crashes.
