---
title: Systems and Processes
---
> [!ABSTRACT]
> 
> This module explores how a program transitions from a static executable file to a living "process" managed by the Operating System. We focus on the life cycle of a process, the boundaries of memory isolation, and the mechanics of the "Fork-Exec-Wait" loop—the foundational logic of every Command Line Interface (CLI).

### 1. The Process Life Cycle
Understanding how the OS creates, tracks, and destroys execution units.
- **[[Process of Operating System]]**: Introduction to the Process Control Block (PCB), Process IDs (PIDs), and the states of execution (Running, Waiting, Terminated).
- **Process Creation**: Mastering the `fork()` system call to create exact duplicates of a running program.

---
### 2. Memory Isolation & Virtualization
How the OS keeps processes from interfering with one another.
- **[[Virtual Memory]]**: The architecture of address spaces. Understanding why parent and child processes see the same addresses but different data.
- **Copy-on-Write (CoW)**: The optimization trick that makes `fork()` efficient by sharing physical RAM until a write occurs.

---
### 3. Execution & Parsing
The tools required to build a functional Shell.
- **[[String Tokenize]]**: Using `strtok` and pointer arrays to transform a user's command string (e.g., `"ls -l"`) into a format that the system can execute (`char *argv[]`).
- **The Exec Family**: Replacing the current process image with a new program (e.g., turning a child process into `grep` or `gcc`).

---
### 4. Practical Roadmap: The "Fork-Exec-Wait" Loop
Every shell follows this specific pattern, utilizing the concepts in this directory:
1. **Read**: Get a string from the user.
2. **Tokenize**: Use **[[String Tokenize]]** to split the command and arguments.
3. **Fork**: Use **[[Process of Operating System]]** logic to create a child.
4. **Exec**: In the child, use an `exec` call to start the command.
5. **Wait**: In the parent, use `wait()` to pause until the child finishes.

---
### Quick Reference: System Call Summary

|**System Call**|**Purpose**|**Primary File**|
|---|---|---|
|`fork()`|Create a child process|[[Process of Operating System]]|
|`wait()`|Wait for child to exit|[[Process of Operating System]]|
|`execvp()`|Load and run a new program|[[String Tokenize]] / [[Process of Operating System]]|
|`mmap()`|Map memory/files|[[Virtual Memory]]|
