---
description: "The application-to-kernel interface mechanism using software traps, descriptor handles, and register parameter passing."
aliases:
  - System Calls
  - Syscall Mechanism
  - Software Traps
  - System Call Interface
tags:
  - cse120
  - operating-systems
  - kernel
  - system-calls
  - API
---
> [!abstract] Abstract
> A **System Call** is the official application programming interface (API) between a user application and the operating system. Because user applications run in unprivileged User Mode, they invoke system calls using software trap instructions to safely request privileged operations from the kernel.
> 
> - **Category:** OS Interface & API Architecture
> - **Core Categories:** Process, Memory, File, Device, and Communication Management.
> - **Key Mechanism:** Software Trap Instruction (`syscall`, `sysenter`, or `int 0x80`).

---

# System Call Categories

The operating system exports a defined set of system calls, accessible via standard C libraries (e.g., `POSIX`, `Win32`):

![[Pasted image 20260720164020.png]]

*   **Process Management:** `fork()`, `exec()`, `exit()`, `wait()`
*   **Memory Management:** `mmap()`, `brk()`, `sbrk()`
*   **File Management:** `open()`, `read()`, `write()`, `close()`
*   **Device Management:** `ioctl()`, `read()`, `write()`
*   **Communication & Inter-Process Communication (IPC):** `socket()`, `bind()`, `connect()`, `pipe()`

---

# The System Call Trap Mechanism

User programs cannot call kernel C function pointers directly. Instead, they trigger a deliberate software exception (a **trap**) that switches execution to Kernel Mode:

![[Pasted image 20260720164244.png]]

### Execution Sequence (`read()` Example)
1.  **API Invocation:** The user application calls `read(fd, buffer, n)`.
2.  **Trap Setup:** The standard C library puts the system call identifier for `read()` into a specific CPU register (e.g., `%eax` or `%rax`).
3.  **Software Trap:** The library executes a hardware trap instruction (e.g., `syscall` or `sysenter`).
4.  **Mode Switch & Dispatch:** The CPU switches to **Kernel Mode**, saves user register states, and jumps to the kernel's central `SyscallHandler` via the interrupt vector table.
5.  **Service Execution:** The kernel reads the system call ID from the register, validates arguments, and executes `sys_read()`.
6.  **Return To User Space:** Upon completion, the kernel writes the return value to a register and executes a return-from-trap instruction (`sysret`), which restores user registers, sets the mode bit back to User Mode ($1$), and resumes execution.

---

# Referencing Kernel Objects: Handles vs. Pointers

Because user processes and the OS kernel operate in separate memory address spaces, applications cannot pass raw kernel memory pointers to system calls.

> [!important] Safe Object Descriptors
> The kernel uses integer **handles** or **descriptors** (e.g., Unix File Descriptors like `fd = 3`) instead of direct pointers. The kernel indexes these integers into private process lookup tables, ensuring applications cannot forge pointers to corrupt kernel memory structures.

---

# Related Notes

- [[Operating Systems/Kernel & Architecture/Dual-Mode Operation & Memory Protection|Dual-Mode Operation & Memory Protection]]
- [[Operating Systems/Kernel & Architecture/Interrupts and Exceptions|Interrupts and Exceptions]]
- [[Operating Systems/Kernel & Architecture/Process/index|Process Subsystem]]
- [[Operating Systems/Kernel & Architecture/Thread/index|Thread Subsystem]]
