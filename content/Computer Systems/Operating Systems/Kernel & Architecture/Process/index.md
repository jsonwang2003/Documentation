---
title: Process Management
description: A directory covering process abstractions, execution states, PCB structures, lifecycle system calls (fork/exec/wait/exit), and IPC mechanisms.
aliases:
  - Process Management Directory
  - Process Hub
  - Process Index
  - Process
tags:
  - index
  - operating-systems
  - processes
---
> [!abstract] Overview
> The **Process** is the primary abstraction used by an operating system to manage execution, virtualize hardware CPU cores, enforce memory boundaries, and allocate system resources. This module covers process execution structures, state transitions, lifecycle APIs, and inter-process communication models.

---

# Core Module Notes

| Note Link                                                                                                        | Description                                                                                                                                                                       | Key Primitives & APIs                                      |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **[[Process Abstraction & PCB\|Process Abstraction & PCB]]**     | Details the distinction between programs and processes, address space memory layouts (Text, Data, Heap, Stack), execution state transitions, and the Process Control Block (PCB). | `PCB`, `task_struct`, Address Space, Context Switching     |
| **[[Process Lifecycle & API\|Process Lifecycle & API]]**         | Covers process creation models (`fork` + `exec` vs `CreateProcess`), parent-child process hierarchies, termination mechanics, and zombie/orphan management.                       | `fork()`, `exec()`, `wait()`, `exit()`, Zombie, Orphan     |

---

# Process API Quick Reference

```mermaid
flowchart TD
    PARENT["<b>Parent Process</b>"]
    FORK["<b>fork()</b><br/><i>Clones process (returns twice: PID to parent, 0 to child)</i>"]

    PARENT_EXEC["<b>Parent Execution</b>"]
    CHILD_EXEC["<b>Child Execution</b>"]

    WAIT["<b>wait()</b><br/><i>Blocks until child exits</i>"]

    EXEC["<b>exec()</b><br/><i>Overwrites address space with new program</i>"]
    EXIT["<b>exit()</b><br/><i>Terminates & sets status code</i>"]

    PARENT --> FORK
    FORK --> PARENT_EXEC
    FORK --> CHILD_EXEC

    PARENT_EXEC --> WAIT
    CHILD_EXEC --> EXEC
    EXEC --> EXIT

    EXIT -->|Kernel releases zombie PCB| WAIT

    classDef cellStyle font-size:15px,padding:12px;
    class PARENT,FORK,PARENT_EXEC,CHILD_EXEC,WAIT,EXEC,EXIT cellStyle
```

---

# Related Sections

- [[Dual-Mode Operation & Memory Protection|Dual-Mode Operation & Memory Protection]]
- [[System Calls|System Calls]]
- [[Interrupts and Exceptions|Interrupts and Exceptions]]
- [[Computer Systems/Operating Systems/index|Operating Systems Main Index]]