---
title: Practice Examples
---
> [!ABSTRACT]
> 
> This repository is a comprehensive guide to systems-level development in C. It tracks the journey from fundamental bit manipulation and memory addressing to the construction of complex systems like custom shells, memory allocators, and web servers.

## 1. [[index 1|Lecture Examples (Theory)]]
The foundational concepts explained through lecture notes and diagrams.
- **[[Computer Systems/System Programming/Intro to Systems Programming/index|Intro to Systems Programming]]**: Hashing, pointer fundamentals, and memory sizing.
- **[[Computer Systems/System Programming/Strings and Data/index|Strings and Data]]**: Bitwise logic, signedness, and the UTF-8/Unicode standard.
- **[[Computer Systems/System Programming/Systems and Processes/index|Systems and Processes]]**: OS process lifecycles, virtual memory, and command tokenization.
- **[[Computer Systems/System Programming/Memory Management/index|Memory Management]]**: Heap mechanics, struct alignment, and custom allocator implementation.
- **[[Computer Systems/System Programming/Digital Communication/index|Digital Communication]]**: Network I/O and the HTTP request/response cycle.

---
## 2. Problem Sets (Reinforcement)
Practical exercises categorized by topic to bridge the gap between lecture and projects.
- **[[Computer Systems/System Programming/coursework/Problem Sets 1/index|PS 1: Data & Bit Manipulation]]**: Exercises in binary conversion, bitwise flags, and UTF-8 handling.
- **[[Computer Systems/System Programming/coursework/Problem Sets 2/index|PS 2: I/O and Security]]**: Working with standard input, hexadecimal converters, and SHA256 hashing.
- **[[Computer Systems/System Programming/coursework/Problem Sets 3/index|PS 3: OS & Shell Logic]]**: Implementing `fork`, `exec`, argument parsing, and file system navigation.
- **[[Computer Systems/System Programming/coursework/Problem Sets 4/index|PS 4: Allocator Mechanics]]**: Logic for pointer arithmetic, block headers, coalescing, and memory alignment.
- **[[Computer Systems/System Programming/coursework/Problem Sets 5/index|PS 5: Network & Web Servers]]**: Building echo servers, URL decoders, and HTTP path recognizers.

---
## 3. [[Computer Systems/System Programming/coursework/Program Assessments/index|Program Assessments (Mastery)]]
Major projects that synthesize all previous concepts into functional system applications.

| **Project**                                                            | **Key Concepts**                                                                                |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **[[Computer Systems/System Programming/coursework/Program Assessments/UTF-8\|UTF-8]]**                               | Bitwise logic, variable-width encoding, multi-byte parsing (stripping headers vs. payload).     |
| **[[The Pioneer Shell\|The Pioneer Shell]]**       | The **Fork-Exec-Wait** loop, process management, signal handling, and environment variables.    |
| **[[Malloc\|Malloc]]**                             | **The Heap**, explicit free lists, block metadata (headers/footers), alignment, and coalescing. |
| **[[Web Server\|Web Server]]**                     | **TCP Sockets**, HTTP protocol parsing (GET/POST), state management, and path routing.          |
| **[[Hashing and Passwords\|Hashing & Passwords]]** | Data integrity, **SHA256**, salting, and binary-to-hexadecimal conversion.                      |

---
## Learning Roadmap
1. **Bit-Level Mastery**: Start with `Strings and Data` and `PS 1` to understand how the CPU sees information.
2. **Pointer & Process Control**: Study `Systems and Processes` and `PS 3` to learn how the OS runs programs.
3. **Advanced Resource Management**: Tackle `Memory Management` and `PS 4` to understand the Heap.
4. **Network Integration**: Use `Digital Communication` and `PS 5` to scale your programs to the web.