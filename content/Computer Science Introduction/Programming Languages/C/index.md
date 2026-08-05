---
title: C
---
> [!ABSTRACT] 
> 
> This repository is a comprehensive guide to **C Programming**, ranging from basic syntax to low-level **Systems Programming**. It covers memory management, bitwise manipulation, network communication, and the implementation of core system utilities.

---
## Foundational Toolkits
These core modules provide the essential rules and conceptual frameworks needed before tackling complex system implementations:
- **[[Computer Science Introduction/Programming Languages/C/Syntax/index|Syntax and Basics]]**: The entry point for C. Covers data types, arrays, control flow, and standard I/O.
- **[[Computer Systems/System Programming/index|Lecture Examples]]**: Conceptual deep dives into how the Operating System manages processes, pointers, and memory headers.

---
## Curriculum Map
### [[Computer Systems/System Programming/Memory Management/index|Memory Management]]
Focuses on the lifecycle of data in memory and the mechanics of the stack and heap.
- Explores **manual memory allocation**, the implementation of custom allocators, and the prevention of resource leaks.
- Covers the structural layout of data using **Structs** and the abstraction of **Virtual Memory**.
### [[Computer Systems/System Programming/Strings and Data/index|Strings and Data]]
Examines how high-level information is translated into binary representations.
- Bridges the gap between **Signed/Unsigned numbers** and character encodings like **UTF-8**.
- Focuses on the low-level manipulation of data through bitwise operators and string buffers.
### [[Computer Systems/System Programming/Digital Communication/index|Digital Communication]]
Introduction to how isolated systems exchange data across networks.
- Covers the basics of the **HTTP protocol** and local socket communication.
- Introduces data integrity and security through **Cryptographic Hashing**.

---
## Knowledge Map (Problem Sets)
## [[Computer Systems/System Programming/coursework/Problem Sets 1/index|Problem Set 1: Bits and Encodings]]
- Focuses on **low-level bit manipulation** and the mechanics of the **UTF-8** variable-width encoding standard.
- Essential for understanding how Unicode characters are serialized into byte streams and manipulated via bitmasks.
## [[Computer Systems/System Programming/coursework/Problem Sets 2/index|Problem Set 2: Standard I/O and Hashing]]
- Focuses on **command line arguments**, **stream processing**, and data integrity.
- Involves building utilities that interact with standard input/output and implement cryptographic hashing for security.
## [[Computer Systems/System Programming/coursework/Problem Sets 3/index|Problem Set 3: Processes and Shells]]
- Focuses on **System Calls** and the fundamentals of building a command-line interface.
- Explores process creation, execution of external commands, and the parsing of user input into executable arguments.
## [[Computer Systems/System Programming/coursework/Problem Sets 4/index|Problem Set 4: The Heap Allocator]]
- Focuses on the **internals of manual memory management** and the implementation of a custom `malloc` library.
- Covers block metadata, fragmentation strategies like splitting/coalescing, and the use of double pointers for heap navigation.
## [[Computer Systems/System Programming/coursework/Problem Sets 5/index|Problem Set 5: Web Servers]]
- Focuses on **Socket Programming** and the implementation of the **HTTP protocol**.
- Covers the creation of network listeners, parsing request paths, and serving dynamic content over TCP/IP connections.

---
## Program Assessments
Final projects that integrate all learned concepts into functional systems.
- **[[Computer Systems/System Programming/coursework/Program Assessments/UTF-8|UTF-8]]**: A robust encoder and decoder for Unicode characters.
- **[[Hashing and Passwords|Hashing and Passwords]]**: Implementing secure data verification and password storage logic using SHA256.
- **[[The Pioneer Shell|The Pioneer Shell]]**: A custom-built Unix shell featuring process control and command parsing.
- **[[Malloc|Malloc]]**: A custom heap memory manager implementing dynamic allocation and freeing logic.
- **[[Web Server|Web Server]]**: A multi-functional HTTP server capable of handling network requests and serving content.