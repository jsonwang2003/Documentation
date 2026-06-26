---
title: Intro to Systems Programming
---
> [!ABSTRACT]
> 
> This chapter explores the interface between C code and the Operating System. It covers how the OS manages running programs through processes, how memory is structured in the address space, and the mechanics of pointers and data hashing.
### 1. Data Integrity & Security
Methods for identifying, verifying, and parsing raw data within a system.
- **[[Hashing (SHA256)]]**: Using deterministic, one-way functions to create unique "fingerprints" for data like Git commits and SSH keys.

---
### 2. Memory Mechanics & Pointer Logic
Understanding how C interacts with the physical and logical layout of memory.
- **[[Pointers and Reference]]**: Managing memory addresses and the necessity of passing explicit lengths for non-null-terminated arrays.
- **[[Returning Pointer]]**: The dangers of "Dangling Pointers" and why returning addresses of stack-allocated local variables leads to memory corruption.
- **[[Sizeof]]**: Determining the byte-footprint of data types and understanding why all pointers share the same size in a 64-bit address space.

---
### 3. Concept Summary Table

|**Term**|**Summary**|
|---|---|
|**Address Space**|The total memory allocated to a process by the OS.|
|**Stack**|Region for local variables; grows toward lower addresses as shown by the Stack Pointer (SP).|
|**PID**|A unique ID the OS uses to track each active process.|
|**Token**|A chunk of text separated by delimiters, often parsed from a command line.|