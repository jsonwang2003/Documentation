---
description: How the OS protects itself from applications, and applications from each other, using CPU-enforced kernel/user modes and privileged instructions.
tags:
  - Operating-System
  - protection
aliases:
  - Kernel Mode vs. User Mode
  - Privileged Instructions
  - Dual-Mode Operation
---
> [!abstract] Purpose 
> How can the OS perform special tasks such as managing resources? How can the OS protect itself from applications, and protect applications from each other? Without an answer to this, any user program could directly control hardware, read or overwrite any other program's memory, or disable the OS entirely — there would be no real operating system, just a pile of code all running with equal, unrestricted power.
> 
> - **Category:** Process Management / Protection
> - **Solves:** gives the OS a hardware-enforced way to be **privileged** — able to do things ordinary programs can't — so it can safely manage resources shared by many mutually-distrusting programs.
> - **Typical use cases:** foundational to every general-purpose multi-tasking OS kernel; almost everything else in an OS (scheduling, memory management, I/O, [[System Calls]]) depends on this protection boundary existing.

---

# Concepts

## Dual-Mode Operation

Every CPU core can run in one of two modes:

- **Kernel Mode:** can run _all_ instructions.
- **User Mode:** can only run _non-privileged_ instructions.

The current mode is indicated by a **mode bit** in a protected CPU control register.

## Privileged Instructions

A subset of CPU instructions that can only run in Kernel Mode. The CPU itself checks the mode bit whenever a privileged instruction is executed — attempts to execute one from User Mode are detected and prevented **by the CPU hardware**, not just by convention or software discipline.

## Memory Protection

The OS must protect itself from user programs, and must be able to protect programs from each other. It may or may not protect user programs from the OS itself. This is provided by memory-management hardware — [[Page Table|page table pointers]], page protection, [[Segmentation|segmentation]], and the [[Translation Lookaside Buffer (TLB)|TLB]] — all of which are manipulated using privileged instructions.

---

# How It Works

> [!tip] Key Idea 
> The mode bit is itself protected — it can only be changed via a controlled, privileged mechanism, never flipped directly by a user program. This closes the loop: a user program can't simply grant itself Kernel Mode access, it has to go through a gate the OS controls (see [[System Calls]] and [[Traps and Interrupts]]).

## What Privileged Instructions Can Do

- **Directly access I/O devices** (disk, network, etc.) — restricted for security and fairness, so one program can't monopolize or snoop on a shared device.
- **Manipulate memory-management state** — e.g. page table pointers — preventing an application from accessing another application's (or the OS's) memory.
- **Manipulate protected control registers** — e.g. the mode bit itself — preventing an application from granting itself privileges it shouldn't have.

## The Software of a Typical (Unix) System

![[Pasted image 20260713224011.png]]

A typical Unix-style system is layered around this exact boundary: hardware and the kernel (which can freely use privileged instructions) sit below the line; user-space programs, utilities, and shells sit above it, only ever able to reach privileged functionality by crossing through a controlled interface (see [[System Calls]]).

## A Subtlety: Root Privilege Is Not Kernel Mode

A root (superuser) process has special _privileges_ within the OS's own permission system, but that doesn't mean it can run privileged CPU instructions directly. Root's extra power is a **software-level** concept enforced by the OS (e.g. bypassing file-permission checks); Kernel Mode is a **hardware-level** concept enforced by the CPU itself. Even root-owned processes still run in User Mode day to day, and still have to go through the OS (via [[System Calls]]) to do anything that genuinely requires Kernel Mode.

---

# Algorithm / Example

## Worked Example: A User Program Attempts a Privileged Instruction

1. A user program, running in **User Mode**, issues an instruction that directly accesses the disk (a privileged instruction).
2. The CPU checks the mode bit **before** executing the instruction — sees User Mode, sees the instruction is privileged, and refuses to execute it.
3. Instead of executing, the CPU raises a hardware trap/exception, transferring control to a pre-registered OS handler — this handler runs in **Kernel Mode**.
4. The OS handler decides how to respond: it might terminate the offending program (if this was a genuine violation), or — far more commonly — recognize this as a legitimate request routed through the [[System Calls|system call]] mechanism, perform the disk access on the program's behalf, and return control to User Mode with the result.

> [!note] 
> The precise mechanics of that hand-off — traps, interrupts, and the system call interface itself — are substantial enough to be their own notes: see [[System Calls]] and [[Traps and Interrupts]].

### Trade-offs

- **Time/Space cost:** every crossing of the User/Kernel boundary (a _mode switch_) has real overhead — saving and restoring registers, validating the request — distinct from (but related to) a full [[Context Switching|context switch]] between processes.
- **Fairness / Starvation:** not really applicable to this mechanism directly — dual-mode operation is what _makes_ fair resource arbitration possible elsewhere (e.g. in [[Scheduling]]), rather than being a scheduling policy itself.
- **Why hardware enforcement matters:** because the check happens in the CPU itself, this is a _hard_ security boundary — a buggy or malicious user program cannot simply choose to ignore it, unlike a purely software-convention-based protection scheme.

## Kernel Mode vs. User Mode, Side by Side

| |Kernel Mode|User Mode|
|---|---|---|
|Can execute privileged instructions?|Yes|No — CPU blocks the attempt|
|Can directly access I/O devices?|Yes|No — must go through [[System Calls]]|
|Can modify page table pointers?|Yes|No|
|Can change the mode bit itself?|Yes|No|
|Who runs here?|The OS kernel|Ordinary applications (even root-owned ones)|

---

# Related Notes

- [[System Calls]] — the controlled gateway between User Mode and Kernel Mode.
- [[Traps and Interrupts]] — the hardware mechanism that transfers control to the OS.
- [[Page Table]] / [[Virtual Memory]] — the memory-protection hardware privileged instructions manipulate.
- [[Segmentation]] / [[Translation Lookaside Buffer (TLB)]] — other memory-management hardware mentioned above.
- [[Context Switching]] — the related but distinct cost of switching between processes, not just modes.
- [[Scheduling]] — one of the things dual-mode operation makes possible: fair, protected sharing of the CPU.