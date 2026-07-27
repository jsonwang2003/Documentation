---
description: "The OS process abstraction, address space components, process execution states, and the Process Control Block (PCB)."
aliases:
  - Process Abstraction
  - Process Control Block
  - PCB
  - Process States
  - Process vs Program
tags:
  - operating-systems
  - kernel
  - processes
  - architecture
---
> [!abstract] Abstract
> A **Process** is the fundamental operating system abstraction for a running program. While a program is a passive collection of instructions on disk, a process is an active instance executing in memory. The kernel virtualizes CPU execution and memory by managing process state transitions and tracking execution contexts inside a **Process Control Block (PCB)**.
> 
> - **Category:** OS Kernel Primitives
> - **Primary Responsibilities:** CPU virtualization, state tracking, resource encapsulation.
> - **Key Data Structure:** Process Control Block (PCB / `task_struct` in Linux).

---

# 1. Process vs. Program

*   **Program:** A passive entity residing on disk (an executable file containing machine code instructions, static data, and metadata).
*   **Process:** An active entity residing in memory representing an ongoing execution instance of a program.

![[Pasted image 20260720173254.png]]

A single program can give rise to multiple distinct processes running simultaneously (e.g., opening multiple browser windows or running multiple shell instances).

---

# 2. Components of a Process

A process encapsulates all hardware and software state required to execute a program:

1.  **Memory Address Space:** The virtual memory region allocated to the process.
    *   **Text (Code):** Executable machine instructions.
    *   **Data Segment:** Initialized global and static variables.
    *   **BSS Segment:** Uninitialized global and static variables.
    *   **Heap:** Dynamically allocated memory requested at runtime (e.g., `malloc()`, `new`). Grows upward.
    *   **Stack:** Manages function call frames, local variables, and return addresses. Grows downward.
2.  **Hardware Context:** Registers representing current CPU execution state.
    *   **Program Counter (PC):** Holds the memory address of the next instruction to execute.
    *   **Stack Pointer (SP):** Points to the top of the active execution stack.
    *   **General Purpose Registers:** Contain active temporary variables and calculation operands.
3.  **OS Resource Identifiers:** Open file descriptors, network socket handles, user/group security IDs (UID/GID), and inter-process communication handles.

![[Pasted image 20260714095236.png]]

---

# 3. Process Execution States

At any point in time, a process resides in one of three core execution states:

*   **Running:** The process is currently executing instructions on a physical CPU core.
*   **Ready:** The process is ready to execute but is waiting to be assigned a CPU core by the OS scheduler.
*   **Waiting (Blocked):** The process cannot execute until an external event completes (e.g., I/O operation, timer, or signal arrival).

![[Pasted image 20260714095926.png]]

### State Transition Mechanics
*   **Admitted $\to$ Ready:** The process is created and loaded into memory, ready for scheduling.
*   **Ready $\to$ Running:** The CPU Scheduler selects the process and dispatches it onto an available CPU core.
*   **Running $\to$ Ready:** The OS preempts the running process (e.g., via a periodic timer interrupt) to give CPU time to another process (**Time-Sharing**).
*   **Running $\to$ Waiting:** The process requests an operation that requires waiting (e.g., `read()` from disk) and yields the CPU.
*   **Waiting $\to$ Ready:** The awaited external event completes (e.g., disk I/O interrupt fires), moving the process back to the run queue.

---

# 4. The Processing Illusion & The PCB

The OS provides every process with the illusion that it owns a dedicated CPU. In reality, a single physical CPU core is shared among many processes using **Time-Sharing** driven by periodic hardware timer interrupts.

### The Process Control Block (PCB)
To pause and resume processes seamlessly without modifying application code, the OS kernel maintains a dedicated tracking data structure for every active process called the **Process Control Block (PCB)** (e.g., `struct task_struct` in Linux).
- Contains all of the information about a process
- Memory management information
- Scheduling and execution information
- I/O and file management

When the OS switches execution from Process A to Process B:
1. It saves Process A's register state, PC, and stack pointer into Process A's PCB.
2. It reloads Process B's saved register state, PC, and stack pointer from Process B's PCB into the CPU.
3. It updates the CPU's memory page table pointer to Process B's address space.

---

# Related Notes

- [[Operating Systems/Kernel & Architecture/Process/Process Lifecycle & API|Process Lifecycle & API]]
- [[Operating Systems/Kernel & Architecture/Dual-Mode Operation & Memory Protection|Dual-Mode Operation & Memory Protection]]
- [[Operating Systems/Kernel & Architecture/Interrupts and Exceptions|Interrupts and Exceptions]]