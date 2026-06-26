# Challenges
1. **Performance**: How can we implement virtualization without adding excessive overhead to the system?
2. **Control**: How can we run processes efficiently while retaining control over the CPU?
	- Particularly important to the OS as it is in charge of resources
	- Without **Control**
		- A process could simply run forever and take over the machine
		- A process could access information that is should not be allowed to access
	- The central challenge in building an operating system

> [!Question] How to efficiently virtualize the CPU with Control
> The OS must virtualize the CPU in an efficient manner, but while **retaining control over the system**. To do so, both **hardware** and **operating systems** support will be required. The OS will often use a *judicious bit of hardware support* in order to accomplish its work effectively
> 

# 1. Limited Direct Execution
"Direct Execution": just run the program directly on the CPU

## Basic direct execution protocol **without limits**

| OS                                | Program                        |
| --------------------------------- | ------------------------------ |
| Create Entry for process list     |                                |
| Allocate memory for program       |                                |
| Load program into memory          |                                |
| Set up stack with `argc` / `argv` |                                |
| Clear registers                   |                                |
| Execute **call** `main()`         |                                |
|                                   | Run `main()`                   |
|                                   | Execute **return** from `main` |
| Free memory of process            |                                |
| Remove from process list          |                                |
## Issues
1. If we just run a program, how can the OS make sure the program doesn't do anything we don't want it to do, while running it efficiently?
2. When we are running a process, how does the OS stop it from running and switch to another process, thus implementing the **time sharing** we require to virtualize the CPU?

### Problem #1: Restricted Operations
Obvious advantage of being fast: the program runs natively on the hardware CPU and this executes as quickly as one would expect

**Issue**: 
What if the process wishes to perform some kind of restricted operation? 

Operations such as
	- Issuing an I/O request to a disk
	- Gaining access to more system resources (CPU / Memory)

> [!Question] How to Perform Restricted Operations?
> A process must be able to perform I/O and some other restricted operations, but without giving the process complete control over the system. How can the OS and hardware work together to do so?

#### Approach 1: Naive Approach
One approach would simply to **let any process do whatever it wants in terms of I/O and other related operations**
→ Problem!! Prevents the construction of many kinds of systems that are desirable
 
> [!Example] File system that checks permissions before granting access → read or write the entire disk and all protections would be lost
#### Approach 2: Use Protected Control Transfer
The hardware assists the OS by providing different modes of execution.
- **User Mode** 
	→ applications do not have full access to hardware resources
	→ code that runs in user mode is restricted in what it can do
- **Kernel Mode** 
	→ the OS runs in this mode
	→ the OS has access to the full resources of the machine
	→ code that runs can do what it likes, including **privileged operations** such as:
	- issuing I/O requests
	- executing all types of restricted instructions

**Still has a remaining challenge**: What should a user process do when it wishes to perform some kind of privileged operations?
- All modern hardware today provides the ability for user programs to perform a **system call**, allowing the *kernel* to carefully expose certain key pieces of functionality to user programs such as:
	- Accessing File Systems
	- Creating / Destroying Processes
	- Communicating with other Processes
	- Allocating more Memory
- To execute a system call, a program **must** execute a special **"trap"** instruction
	- Jumps into the kernel and raises the privilege level to *kernel mode*
	- Once in the *kernel*, system can now perform whatever privileged operations are needed (if allowed)
	- When finished, the OS calls a special **return-from-trap** instruction, which returns into the calling user program and reduced the privilege level back to *user mode*
- Hardware needs to be careful when executing a **trap**
	- Make sure to save enough of the caller's register state to be able to return correctly when the OS issues **return-from-trap** instruction
		- On x86, the processor will push the program counter, flags, and a few other registers onto **kernel stack**
		- **return-from-trap** pop these values off the stack and resume execution of the user program
- How does the trap know which code to run inside OS
	- The *kernel* does so by setting up a **trap table** at boot time
		- When the machine boots up in *kernel* mode and thus is free to configure machine hardware as need be
	- The OS tell hardware what code to run when certain exception events occur
	- The OS informs the hardware of the locations of these **trap handlers**
	- Once the hardware is informed, it remembers the location of these handlers until the machine is next rebooted, thus the hardware knows what to do when system calls and other exceptional events take place

#### Protected Control Transfer Protocol Summery

| **OS @ run (Kernel Mode)**                                                                                                                                                             | **Hardware**                                                                 | **Program (User Mode)**                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Create entry for process list <br>Allocate memory for program <br>Load program into memory <br>Setup user stack with `argv` <br>Fill kernel stack with reg/PC <br>**return-from-trap** |                                                                              |                                                                 |
|                                                                                                                                                                                        | Restore regs from kernel stack <br>Move to user mode <br>Jump to main        |                                                                 |
|                                                                                                                                                                                        |                                                                              | Run `main()`<br>$\dots$<br>Call system call<br>**trap** into OS |
|                                                                                                                                                                                        | Save regs to kernel stack<br>Move to kernel mode<br>Jump to trap handler     |                                                                 |
| Handle trap<br>  Do work of `syscall`<br>**return-from-trap**                                                                                                                          |                                                                              |                                                                 |
|                                                                                                                                                                                        | Restore regs from kernel stack<br>Move to user mode<br>Jump to PC after trap |                                                                 |
|                                                                                                                                                                                        |                                                                              | $\dots$<br>return from main<br>**trap(via `exit()`)**           |
| Free memory of process<br>Remove from process list                                                                                                                                     |                                                                              |                                                                 |
