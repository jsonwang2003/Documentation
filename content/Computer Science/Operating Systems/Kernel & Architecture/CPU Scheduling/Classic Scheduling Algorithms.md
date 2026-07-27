---
description: "Analysis of foundational scheduling algorithms: First-Come First-Served (FCFS), Shortest Job First (SJF), Shortest Remaining Time to Completion First (SRTCF), Round Robin (RR), and Priority Scheduling."
aliases:
  - Classic Scheduling Algorithms
  - FCFS
  - SJF
  - SRTCF
  - STCF
  - Round Robin
  - Priority Scheduling
tags:
  - operating-systems
  - kernel
  - scheduling
  - algorithms
---
> [!abstract] Abstract
> Classical CPU scheduling algorithms employ distinct heuristics to sequence runnable tasks. **Non-preemptive policies** (FCFS, SJF) execute jobs to completion or until they block, whereas **preemptive policies** (SRTCF, Round Robin, Priority) use timer interrupts to re-evaluate thread selection dynamically. Each policy presents trade-offs between turnaround time, response latency, and starvation risk.
> 
> - **Category:** Scheduling Policy Algorithms
> - **Provably Optimal Policy:** SRTCF (minimizes average turnaround time).
> - **Interactive Standard:** Round Robin (minimizes response time).

---

# 1. First-Come, First-Served (FCFS / FIFO)

Processes are dispatched in the strict order of their arrival time. FCFS is **non-preemptive**.

![[Pasted image 20260723163520.png]]

### Turnaround Time Example
Consider four jobs arriving at $T=0$ with execution lengths: $J_1=4$, $J_2=4$, $J_3=1$, $J_4=7$.

![[Pasted image 20260723163819.png]]

$$\text{Avg Turnaround Time} = \frac{4 + 8 + 9 + 16}{4} = 9.25\text{ s}$$

If job arrival order changes to $J_1=7, J_2=4, J_3=1, J_4=4$:

![[Pasted image 20260723163913.png]]

$$\text{Avg Turnaround Time} = \frac{7 + 11 + 12 + 16}{4} = 11.5\text{ s}$$

*   **Pros:** Simple, fair arrival ordering, zero starvation.
*   **Cons:** **Convoy Effect**—short jobs get stuck behind a long CPU-bound job, causing high average turnaround time.

---

# 2. Shortest Job First (SJF)

SJF runs the runnable job with the shortest total burst time first. It is **non-preemptive**.

![[Pasted image 20260723164437.png]]

For job lengths $7, 3, 1, 6$ arriving simultaneously:
$$\text{Avg Turnaround Time} = \frac{1 + 4 + 7 + 17}{4} = 7.25\text{ s}$$

*   **Pros:** Provably minimizes average turnaround time *if all jobs arrive simultaneously*.
*   **Cons:** Cannot preempt a long job that started right before short jobs arrive; requires knowing job execution runtimes in advance; risks **starving** long jobs.

---

# 3. Shortest Remaining Time to Completion First (SRTCF / STCF)

SRTCF is the **preemptive** variant of SJF. Whenever a new job arrives, the scheduler compares its remaining execution time against the currently running job. If the new job requires less time, the active job is preempted.

![[Pasted image 20260723174748.png]]

$$\text{Avg Turnaround Time} = \frac{16 + 5 + 1 + 5}{4} = 6.75\text{ s}$$

*   **Pros:** **Provably optimal**—yields the absolute minimum average turnaround time for any workload.
*   **Cons:** Requires advance knowledge of remaining execution times; causes **starvation** for long jobs under heavy short-job workloads.

---

# 4. Round Robin (RR)

Round Robin is a **preemptive** time-sharing algorithm designed for interactive workloads. The scheduler runs each job for a fixed time slice (**quantum**), moving preempted jobs to the back of a circular ready queue.

![[Pasted image 20260723174936.png]]

![[Pasted image 20260723174955.png]]

### Quantum Sizing Trade-off
*   **Quantum too large ($\to \infty$):** RR degrades into non-preemptive FCFS, causing poor response times.
*   **Quantum too small ($\to 0$):** CPU spends all its time context switching, causing massive overhead and low CPU utilization.

*   **Pros:** Excellent interactive **response time**, fair CPU sharing, no starvation.
*   **Cons:** High context-switching overhead; poor average turnaround time when jobs have equal burst lengths.

---

# 5. Priority Scheduling

Each job is assigned a priority integer. The scheduler always dispatches the runnable job with the highest priority (using FIFO to break ties). Can be preemptive or non-preemptive.

*   **Priority Assignment:**
    *   *Internal:* Assigned automatically by the OS based on memory requirements, open file count, or I/O burst ratios.
    *   *External:* Assigned manually by administrators or users (e.g., Unix `nice` values).
*   **Primary Deficit:** **Starvation**—low-priority jobs may wait indefinitely if high-priority jobs continually arrive.
*   **Solution:** **Aging**—gradually increase the priority of jobs that wait in the ready queue for long periods.

---

# 6. Algorithm Comparison Matrix

| Algorithm | Preemptive? | Primary Optimization Goal | Main Advantage | Main Disadvantage |
|---|---|---|---|---|
| **FCFS** | No | Simplicity / Fairness | Simple; no starvation | Convoy effect; high turnaround time |
| **SJF** | No | Average Turnaround Time | Minimizes turnaround time | Requires knowing future runtimes |
| **SRTCF** | **Yes** | Absolute Turnaround Time | **Provably optimal** turnaround | Starves long jobs; requires runtime estimates |
| **Round Robin** | **Yes** | **Response Time** | Fast interactive response; no starvation | Context switch overhead; poor turnaround |
| **Priority** | Both | Policy / Importance | Flexible task prioritization | **Starvation** of low-priority tasks |

---

# Related Notes

- [[Operating Systems/Kernel & Architecture/CPU Scheduling/CPU Scheduling Fundamentals & Metrics|CPU Scheduling Fundamentals & Metrics]]
- [[Operating Systems/Kernel & Architecture/CPU Scheduling/Multilevel Feedback Queue & Real-World Schedulers|Multilevel Feedback Queue & Real-World Schedulers]]
- [[Operating Systems/Kernel & Architecture/Thread/Thread Context Switch & Scheduling|Thread Context Switch & Scheduling]]