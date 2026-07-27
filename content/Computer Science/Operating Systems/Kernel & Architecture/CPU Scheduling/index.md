---
title: CPU Scheduling Index
description: A directory covering CPU scheduler abstractions, scheduling metrics, classical policies (FCFS, SJF, SRTCF, RR, Priority), MLFQ, and production OS schedulers.
aliases:
  - CPU Scheduling Directory
  - Scheduling Index
  - CPU Scheduling Hub
  - CPU Scheduling
tags:
  - index
  - operating-systems
  - kernel
  - scheduling
---
> [!abstract] Overview
> **CPU Scheduling** is the core subsystem responsible for multiplexing physical CPU cores across runnable processes and threads. By separating the **Mechanism** (context switching and state queues) from the **Policy** (selecting which thread runs next), the CPU scheduler balances system performance, responsiveness, resource utilization, and fairness across diverse workloads.

---

# Module Structure & Notes

| Note Link | Description | Key Concepts & Metrics |
|---|---|---|
| **[[Operating Systems/Kernel & Architecture/CPU Scheduling/CPU Scheduling Fundamentals & Metrics\|CPU Scheduling Fundamentals & Metrics]]** | Explores policy vs. mechanism, dispatcher triggers, scheduling metrics ($T_{\text{turnaround}}$, $T_{\text{response}}$), workload profiles (batch vs. interactive), CPU utilization math, and starvation. | Turnaround Time, Response Time, Preemption, CPU Utilization, Starvation |
| **[[Operating Systems/Kernel & Architecture/CPU Scheduling/Classic Scheduling Algorithms\|Classic Scheduling Algorithms]]** | Analyzes foundational scheduling policies: First-Come First-Served (FCFS), Shortest Job First (SJF), Shortest Remaining Time to Completion First (SRTCF), Round Robin (RR), and Priority Scheduling. | FCFS, SJF, SRTCF, Round Robin, Quantum, Priority |
| **[[Operating Systems/Kernel & Architecture/CPU Scheduling/Multilevel Feedback Queue & Real-World Schedulers\|Multilevel Feedback Queue & Real-World Schedulers]]** | Details adaptive priority decay in MLFQ, I/O burst handling, and production schedulers (Linux Completely Fair Scheduler - CFS, macOS/Windows MLFQ). | MLFQ, Priority Decay, I/O Bursts, Linux CFS, Windows Scheduler |

---

# Policy vs. Mechanism in CPU Scheduling

```c
void yield() {
    thread_t old_thread = current_thread;
    
    current_thread = get_next_thread();       // <--- POLICY (Which thread runs next?)
    
    append_to_queue(ready_queue, old_thread);
    context_switch(old_thread, current_thread); // <--- MECHANISM (Assembly hardware switch)
    return;
}
```

---

# Related Modules

- [[Operating Systems/Kernel & Architecture/Thread/Thread Context Switch & Scheduling|Thread Context Switch & Scheduling]]
- [[Operating Systems/Kernel & Architecture/Process/Process Abstraction & PCB|Process Abstraction & PCB]]
- [[Operating Systems/Kernel & Architecture/index|Kernel & Architecture Main Directory]]
- [[Operating Systems/index|Operating Systems Main Directory]]