---
title: "Sequential Circuits"
description: "Master index for Sequential Circuits: memory principles, Memory Hierarchy (Flip-Flops, SRAM, DRAM, SSD), Latches & Flip-Flops, Registers & Counters, Finite State Machines (FSMs), and Timing Constraints."
aliases:
  - Sequential Circuit Index
  - Sequential Logic Hub
  - Sequential Systems
tags:
  - index
  - digital-systems
  - sequential-logic
  - memory-hierarchy
  - fsm
---
> [!abstract] Overview
> Unlike combinational networks whose outputs depend solely on present inputs, a **Sequential Circuit** incorporates **memory elements** and feedback paths. Its output and next state depend on both **current inputs** and **past outputs (stored state history)**. Sequential logic enables data storage, execution of sequenced multi-cycle task pipelines, and the physical realization of the **Memory Hierarchy**—spanning ultra-fast 6-Transistor (6T) SRAM cells and Flip-Flops down to high-density capacitor-based DRAM and non-volatile mass storage.

---

## 1. Fundamentals of Sequential Circuits & Memory

Sequential circuits form the core of digital processing units by maintaining state and controlling operations over time.

![[Pasted image 20260824142942.png]]

* **Sequential Logic Function:** Output $Y(t) = f(X(t), S(t))$, where $X(t)$ represents current inputs and $S(t)$ represents internal state variables.
* **Core Capabilities:**
  1. **Data Storage:** Persists single bits or multi-bit binary words across clock cycles.
  2. **Task Sequencing:** Executes ordered sequences of operations (e.g., instruction execution, counter stepping).

---

## 2. Memory Hierarchy & RAM Technology Primitives

System memory is structured in a **Hierarchy** that trades off access latency for density, capacity, and cost per bit.

![[Pasted image 20260824171727.png]]
*SRAM*
### SRAM vs. DRAM Architecture

| Memory Level          | Hardware Technology         |           Cell Architecture           | Operating Characteristics                                                                  |
| --------------------- | --------------------------- | :-----------------------------------: | ------------------------------------------------------------------------------------------ |
| **Registers**         | Flip-Flops                  |         Master-Slave Latches          | Fastest speed, lowest density, volatile.                                                   |
| **Cache**             | **SRAM** *(Static RAM)*<br> |        **6 Transistors (6T)**         | Fast access; retains value continuously as long as power is applied.                       |
| **Main Memory**       | **DRAM** *(Dynamic RAM)*    | **1 Transistor + 1 Capacitor (1T1C)** | High density; charge leaks off capacitor over time, **requiring periodic refresh cycles**. |
| **Secondary Storage** | **Hard Disk / SSD**         |         NAND Flash / Magnetic         | Non-volatile (permanent); slowest access speed, highest storage capacity.                  |


---

## Submodule Notes

- [[Computer Systems/Digital Systems/Sequential Circuit/Latches & Flip-Flops|Latches & Flip-Flops]]  
  Fundamental bistable storage elements: SR Latches, level-sensitive Gated Latches, D Latches, Master-Slave D Flip-Flops, edge-triggering, transistor counts, and reset/preset controls.

- [[Computer Systems/Digital Systems/Sequential Circuit/Registers and Counters|Registers and Counters]]  
  Multi-bit parallel registers, Serial-In/Serial-Out shift registers, sequence pattern recognizers, 4:1 multiplexer-based Universal Shift Registers, and modulo state counters.

- [[Computer Systems/Digital Systems/Sequential Circuit/Finite State Machines|Finite State Machines]]  
  Behavioral controllers, state diagrams, state transition tables, Mealy vs. Moore machines, and the 5-step FSM circuit synthesis process.

- [[Computer Systems/Digital Systems/Sequential Circuit/Timing Constraints in Sequential Designs|Timing Constraints in Sequential Designs]]  
  Physical timing analysis: contamination ($t_{cd}, t_{ccq}$) and propagation ($t_{pd}, t_{pcq}$) delays, setup ($t_{\text{setup}}$) and hold ($t_{\text{hold}}$) constraints, hold violation fixes, and worst-case clock skew ($t_{\text{skew}}$).

---

## Related Submodules & Directories

- [[Computer Systems/Digital Systems/ALU/index|Arithmetic Logic Units (ALUs)]]
- [[Computer Systems/Digital Systems/Logic Design/index|Logic Design & K-Maps]]
- [[Computer Systems/Digital Systems/Number Representation & Basic Logic Gates/index|Number Representation & Basic Logic Gates]]
- [[Computer Systems/Digital Systems/index|Digital Systems Main Index]]
- [[Computer Systems/index|Computer Systems]]