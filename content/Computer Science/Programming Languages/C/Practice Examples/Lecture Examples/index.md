---
title: Lecture Examples
---
> [!ABSTRACT]
> 
> This repository is a structured journey through systems-level development in C. It spans the gap between raw hardware bits and high-level network communication, focusing on how a programmer manages memory, interacts with the Operating System, and builds robust, performance-oriented software.

## 1. [[Intro to Systems Programming/index|Intro to Systems Programming]]
The entry point. This module establishes the basic relationship between the programmer and memory addresses.
- **Core Tools**: Understanding `sizeof`, pointer vs. value references, and the mechanics of SHA256 hashing.
- **Memory Safety**: Why returning pointers from the stack leads to undefined behavior.

---
## 2. [[Strings and Data/index|Strings and Data]]
Understanding the binary representation of information.
- **Bitwise Logic**: Masking and shifting to manipulate data at the lowest level.
- **Interpretation**: The difference between signed/unsigned interpretation and the traps of character signedness.
- **Internationalization**: The variable-width mechanics of **UTF-8** and Unicode.

---
## 3. [[Systems and Processes/index|Systems and Processes]]
The interface between your code and the Operating System kernel.
- **Process Management**: How the OS creates execution units via `fork()`.
- **The Shell Loop**: Tokenizing user input and executing commands using the Fork-Exec-Wait pattern.
- **Virtualization**: The illusion of private memory via **Virtual Memory** and Copy-on-Write.

---
## 4. [[Computer Science/Programming Languages/C/Practice Examples/Lecture Examples/Memory Management/index|Memory Management]]
The deep-dive into the "Heap." Learn how to build the memory management tools that modern languages take for granted.
- **Struct Design**: How padding and alignment affect memory footprint.
- **Custom Allocators**: Implementing `malloc`, `free`, and `realloc` using block metadata and coalescing logic to fight fragmentation.

---
## 5. [[Digital Communication/index|Digital Communication]]
Scaling your C knowledge to talk to other computers.
- **Network I/O**: Sockets, reading/writing across buffers, and handling connection state.
- **HTTP Protocol**: Parsing request paths, managing query parameters, and serving dynamic content.

---
## Technical Mapping Reference

|**Folder**|**Level**|**Hardware Focus**|
|---|---|---|
|**Strings & Data**|Low|ALU / Bit Registers|
|**Intro**|Mid|Memory Addresses|
|**Systems & Processes**|Mid|OS Kernel / CPU Scheduler|
|**Memory Management**|High-Mid|Physical RAM Management|
|**Digital Comm**|High|NIC / Network Stack|

---
### Suggested Learning Path
1. **The Foundations**: [[Intro to Systems Programming/index|Intro]] $\rightarrow$ [[Strings and Data/index|Strings]]
2. **The Environment**: [[Systems and Processes/index|Systems & Processes]]
3. **The Internals**: [[Computer Science/Programming Languages/C/Practice Examples/Lecture Examples/Memory Management/index|Memory Management]]
4. **The Application**: [[Digital Communication/index|Digital Communication]]