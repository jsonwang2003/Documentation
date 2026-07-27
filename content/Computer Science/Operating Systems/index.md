---
title: Operating Systems
---
# What is an Operating System?
- Code that sits between applications and hardware
- Provides abstractions to layers above
- implements abstractions for an manages resources below

## The OS and Hardware
The OS **controls and mediates** access to hardware resources
- Computation (CPUs)
- Volatile storage (memory) and persistent storage (disk, etc)
- Communication (network, etc)
- Devices (keyboard, camera, monitor, etc)
OS operations
- Resources allocation
- Resources reclamation
- Protection ― between and from applications
  
![[Pasted image 20260720001341.png|248]]
*layer of Operating System in Computer Operations*

## OS and Applications
The OS provides **abstractions** to applications
The OS defines a set of logical resources **(objects)** and a set of well-defined operations on those objects **(interfaces)**
- Files ― `create`, `read`, `write`
- Threads ― `create`, `yield`, `exit`
Many benefits over dealing with hardware directly $\begin{cases}\text{protect between from users} \\ \text{set up standardized procedure} \end{cases}$
- Hides the complexity of hardware
- Allows hardware to evolve independently
- Provides the illusion of "infinite memory"
	- or "sole application running"
Users and programs can **safely coexist, cooperate, and share resources**
- Concurrent execution of multiple applications (time sharing)
- Communication among multiple applications (IPC ― sockets, pipes, etc)
- Share common services
	- No need to implement your own file system

## How Similar are Different OSes?
Popular OSes today: Window, Linux, and OS X

![[Pasted image 20260720002052.png]]

## Do OSes Change Over Time?
Core operating systems concepts date back to the 1970s ― have these changed?
Hardware is evolving:
- 60s-70s: mainframes
- 70s-80s: minicomputers
- 80s-90s: PCs
- 90s-00s: laptops
- Today: smartphones, watches, etc
New applications
- Web-based applications
- Virtual Reality
- Machine Learning

---
# What Properties Should an OS Provide?
- Efficiency or Performance?
- Fairness?
- Portability?
- Security?
- Robustness?
- Goals can Conflict
	- Fairness vs. Efficiency
	- Security vs. Performance

## Implications for OS Design
OSes face constraints and tradeoffs
- Different objectives for different applications
OSes must adapt over time due to evolution of
- Hardware
- Applications
Design Principles can guide us
- Abstraction
- Modularity
- Simplicity
- Caching

## Separation of Policy and Mechanism
Fundamental Design Principle
- **Mechanism**: tool that achieves some effect (implementation)
- **Policy**: decision about what effect should be achieved (goal)

CPU scheduling example
- Treat all users equally
- Treat all applications equally
- Prioritize some applications over others

Separation leads to flexibility

# Hardware
![[Pasted image 20260720002852.png]]

## Software of a Typical (Unix) System
![[Pasted image 20260720002942.png]]