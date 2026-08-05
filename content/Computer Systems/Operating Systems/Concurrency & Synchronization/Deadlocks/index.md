---
title: Deadlocks
description: A directory covering deadlock concepts, Coffman conditions, Resource Allocation Graphs, and deadlock strategies (Prevention, Avoidance, Detection, Recovery).
aliases:
  - Deadlocks Directory
  - Deadlock Hub
  - Deadlocks Index
tags:
  - index
  - operating-systems
  - concurrency
  - deadlocks
---
> [!abstract] Overview
> A **Deadlock** occurs when a set of concurrent threads is permanently blocked because every thread in the set is waiting for an event or resource that can only be triggered by another thread in that same set. This module details the four Coffman conditions, Resource Allocation Graphs (RAGs), and system strategies for handling deadlocks.

---

# Core Module Notes

| Note Link | Description | Key Concepts & Primitives |
|---|---|---|
| **[[Deadlock Fundamentals & Coffman Conditions\|Deadlock Fundamentals & Coffman Conditions]]** | Covers formal deadlock definitions, the Dining Philosophers problem, the 4 necessary Coffman conditions, and Resource Allocation Graph (RAG) cycle analysis. | Coffman Conditions, Dining Philosophers, Resource Allocation Graph (RAG), Cycles |
| **[[Deadlock Handling Strategies\|Deadlock Handling Strategies]]** | Details the 4 core deadlock management strategies: Ostrich algorithm, Deadlock Prevention, Deadlock Avoidance (Banker's Algorithm), and Detection & Recovery. | Deadlock Prevention, Total Resource Ordering, Banker's Algorithm, Thread Aborting |

---

# The Four Coffman Conditions

![[Pasted image 20260723151922.png]]

---

# Related Modules

- [[Locks|Locks]]
- [[Critical Sections & Mutual Exclusion|Critical Sections & Mutual Exclusion]]
- [[Computer Systems/Operating Systems/Concurrency & Synchronization/index|Concurrency & Synchronization Main Index]]