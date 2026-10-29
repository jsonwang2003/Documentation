---
title: "Operating Systems Index" 
description: "Root index for Operating Systems — kernel organization, processes and threads, concurrency and synchronization, memory management, file systems, and communication." 
tags:
- CS/index
- CS/operating-systems 
aliases: ["Operating Systems", "OS Index"] 
---

> [!abstract] Overview 
> Root index for the Operating Systems area. Per CSE120's scope, this covers: kernel organization and structure, processes and threads, concurrency and synchronization, memory management, file systems, and communication.

---

# Foundational Concepts

## Why an OS at All?

An OS turns raw hardware — a CPU that just executes instructions, with no notion of "a program" or "fairness" — into a platform that can safely run many mutually-distrusting programs at once. Everything below is either part of that safety mechanism or built on top of it.

## The Privilege Boundary

Every mechanism in this section ultimately depends on the Kernel Mode / User Mode boundary from [[Dual-Mode Operation & Protection]] and the [[Events]] that are the only way to cross it.

---

# Notes in This Section

|Category|Notes|Description|
|---|---|---|
|**Fundamentals and Protection**|[[Dual-Mode Operation & Protection]], [[Events]]|The Kernel/User Mode boundary, privileged instructions, and the exceptions/interrupts that cross it|
|**Process Management**|[[Processes]], [[Process Creation]], [[Process Termination]], [[Inter-Process Communication]]|The process abstraction, its lifecycle, and how processes communicate|
|**Execution and Scheduling**|[[Threads]], [[Context Switching]], [[Threading Models]]|Execution state separated from the process, and how the CPU switches between threads|
|**Concurrency and Synchronization**|[[Race Conditions]]|Why shared state between threads is dangerous — the motivation for synchronization primitives (mechanisms and patterns still to come)|
|**Virtualization**|_(drafted, not yet finalized)_|Hypervisors, virtualizing privileged instructions/events/CPU/memory/I/O, and hardware virtualization support|
|**File Systems**|[[File Systems Index]]|Disk layouts, buffer caches, reliability, and file I/O — see that index for its own module breakdown|

---

# Related Categories

- [[Algorithms]] — scheduling policies and synchronization mechanisms in this section draw on general algorithmic techniques from there.
- [[Digital Systems Index]] — the hardware-level logic (privileged instructions, mode bits, registers) this entire section is built on top of.