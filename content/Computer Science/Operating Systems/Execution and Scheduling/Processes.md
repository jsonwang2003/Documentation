The **process** is the OS abstraction for a running program
- Used to manage execution, scheduling, and other resources
Simplest process: sequential process
- Everything happens sequentially
- One instruction at a time

# Components
A process contains all state for a program in execution
- A memory address space
- code and data for the executing program
- an execution stack encapsulating the state of procedure calls
- The program counter (PC) indicating the next instruction
- A set of registers with current values
- A set of operating system resources
	- Open files, network connections
A process is named using its process ID (PID)

# Process vs. Program
A process is an instance of a program in execution

![[Pasted image 20260714095151.png]]

# Process Address Space
![[Pasted image 20260714095236.png]]

# Process State
A process has an execution state that indicates what it is currently doing
- Running ― executing instructions on the CPU (this process has control of the CPU)
- Ready ― waiting to the assigned to the CPU (Ready to execute, but another process is executing on the CPU)
- Waiting (blocked) ― waiting for an event (it cannot make progress until an event occurs)
As a process executes, it moves from state to state

![[Pasted image 20260714095926.png]]

---
# The Processing Illusion
Every process thinks it owns the CPU
**In reality**:
- With 1 CPU, all processes share the same physical CPU
- With multiple CPUs, processes share the multiple CPUs
How is this possible?
- Timer interrupts (preemptive scheduling) ― stop current process and run another task
- Data structure to hold execution state while not executing
- [[Scheduling policies]]
![[Pasted image 20260714103517.png]]

## Process Data Structure
Many processes "running" simultaneously
How does the OS represent a process in the kernel?
Process Control Block (PCB)
- Contains all of the information about a process
- Memory management information
- Scheduling and execution information
- I/O and file management
If is a **heavyweight** abstraction

---
# Process Creation
Every process is created by another process
- Parent process creates a Child process using a system call
Child inherits some properties from parent
- Unix: process user ID ― children execute with your privileges
After creating a child, the parent may either wait for it to finish its task or continue in parallel