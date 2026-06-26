---
title: Syntax and Semantics
---
## Documentation Modules
Use the links below to navigate through the specific technical modules:
- **[[Getting Started with C Programming]]**: Environment configuration, compiler setup, and the basic execution model.
- **[[Input-Output in C]]**: Technical implementation of standard I/O operations and format specifiers.
- **[[Computer Science/Programming Languages/C/Syntax/Arrays|Arrays]]**: Memory allocation, indexing, and manipulation of contiguous data sets.
- **[[Strings]]**: Character array management and utility functions within the `string.h` library.
- **[[Practice Programs]]**: A repository of implementation examples for logic verification and skill building.

---
## Technical Overview of C Syntax
C is a high-level, statically typed, procedural language that provides low-level memory access. Its syntax is the foundation for many modern languages, including C++, Java, and C#.
### Program Structure
A standard C source file consists of preprocessor directives, global declarations, and functions. The execution life cycle always begins at the `main()` function.

```c
#include <stdio.h> // Header file inclusion for standard functions

int main(void) {    // Program entry point
    // Logic implementation
    return 0;      // Process exit code
}
```

### Data Types and Memory
C requires explicit type declaration. This allows the compiler to determine the exact memory footprint required for each variable.

|**Type**|**Description**|**Size (Platform Dependent)**|
|---|---|---|
|`int`|Integer values|Typically 4 bytes|
|`float`|Single-precision floating point|Typically 4 bytes|
|`double`|Double-precision floating point|Typically 8 bytes|
|`char`|Single character/byte|1 byte|
### Syntactic Requirements
To ensure code is valid and maintainable, adhere to the following structural rules.
- **Statement Termination**: Every statement must conclude with a semicolon (`;`).
- **Case Sensitivity**: The compiler is case-sensitive; `Value` and `value` are distinct identifiers.
- **Comments**: Use `//` for inline documentation and `/* ... */` for block-level documentation.
- **Scope Delimitation**: Use curly braces `{}` to define the scope of functions, loops, and conditional blocks.
### Control Logic
Program flow is governed by two primary categories of control structures:
1. **Selection Statements**: `if`, `else if`, `else`, and `switch` for branching logic.
2. **Iteration Statements**: `for`, `while`, and `do-while` for repetitive execution.

---

> [!Note] Technical Note: 
> C does not automatically initialize local variables. Uninitialized variables contain indeterminate "garbage" data. Always initialize variables before use to prevent undefined behavior.