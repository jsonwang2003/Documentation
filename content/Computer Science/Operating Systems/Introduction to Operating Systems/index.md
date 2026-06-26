---
title: Introduction to Operating Systems
---
## What Happens when a Program Runs?
### Von Neumann Model of Computing
- Many millions of times every second
- The Processor 
	1. **fetches** an instruction from memory
	2. **decodes** it → Figure out what instruction it is
	3. **executes** it → Does what the instruction is supposed to do
	4. moves on to the next instruction until the program completes

## What is Operating System
> [!Abstract] Definition
> A body of **software** that is responsible for making it easy to run programs. It allows programs to:
> - Share Memory
> - Enabling Programs to Interact with Hardware Devices
> - Seemingly run many programs at the same time
>   
> It is in charge of making sure the system operates **correctly** and **efficiently** in an *easy-to-use manner*

### Virtualization
A technique that the OS takes a **physical** resource (processor, memory, disk, etc.) and **transforms** it into a more general, powerful, and easy-to-use **virtual** form of itself
- Provides **"System Calls"** APIs (standard library) to allow users communicate with OS on what to do
	→ makes use of the features of the virtual machine
- Manage resources (CPU, Memory, Disk, etc.)
	→ virtualization allows 
	1. many programs to run (thus sharing the CPU) 
	2. many programs to concurrently access their own instructions and data (thus sharing the memory)
	3. many programs to access devices (thus sharing disks and so forth)

> [!Question] How to Virtualize Memory
> *Why* the OS virtualize memory because it makes the system easier to use
> 
> *How* does OS virtualize memory:
> - What mechanisms and policies are implemented by the OS to attain virtualization?
> - How does the OS do so efficiently?
> - What hardware support is needed?

---
## Table of Contents




