## 1. Threat Modeling & Security Properties

- **Threat Model**: Defines assets (WHAT to protect) and attackers (WHO to protect against, including their capabilities and goals). Defines the problem scope and attack surface.
    
- **Security Properties (CIA +)**:
    
    - **Confidentiality**: Prevention of unauthorized access to information.
        
    - **Integrity**: Prevention of unauthorized modifications or destruction of data.
        
    - **Availability**: Prevention of unauthorized denial of service to legitimate users.
        
    - **Authenticity**: Identification and assurance of the origin of data or communication.
        
    - **Privacy**: Protection of sensitive information and personally identifiable data.
        

## 2. Control Flow & Memory Vulnerabilities

### A. Memory Layout & Stack Frame Structure

- **Stack Layout (High to Low Memory)**: Arguments $\rightarrow$ Return Address (`saved %eip`) $\rightarrow$ Frame Pointer (`saved %ebp`) $\rightarrow$ Stack Canary $\rightarrow$ Local Variables / Buffers.
    
- **Registers**:
    
    - **`%esp` (Stack Pointer)**: Points to the current top of the stack (grows downwards from high to low memory addresses).
        
    - **`%ebp` (Frame Pointer)**: Points to the base of the current stack frame.
        

### B. Vulnerability Types

- **Stack Buffer Overflow**: Writing past the boundaries of a stack buffer overwrites adjacent memory (local variables, saved `%ebp`, and saved `%eip`). Functions like `gets()` and `strcpy()` are unsafe because they do not enforce bounds checking.
    
- **Format String Vulnerabilities**: Occur when user input is passed directly as the format argument to `printf(user_input)`.
    
    - `%x` / `%s`: Reads values from the stack / arbitrary memory addresses.
        
    - `%n`: Writes the number of characters printed so far into a memory location specified by a pointer on the stack.
        
- **Heap Vulnerabilities**:
    
    - **Double Free**: Freeing the same heap memory chunk twice, corrupting heap bin structures (doubly-linked lists).
        
    - **Use-After-Free (UAF)**: Accessing or modifying memory after it has been freed.
        
- **Pointer Subterfuge**: Overwriting target function pointers or data pointers to alter program execution flow without needing to hit the return address directly.
    
- **Integer Overflow**: Arithmetic wraparound leading to unexpectedly small dynamic allocations, causing subsequent heap buffer overflows.
    

## 3. Defense Mitigations & Evasion Techniques

|**Defense Mechanism**|**How It Works**|**Common Bypass Strategy**|
|---|---|---|
|**Stack Canary**|Places a secret value before saved `%ebp`/`%eip`. Checks canary integrity before function return.|**Canary Leak**: Read via format string/info leak.<br><br>  <br><br>**Non-contiguous Write**: Pointer subterfuge.|
|**DEP / $W \oplus X$**|Memory pages are writable OR executable, but never both. Prevents executing injected stack shellcode.|**Return-to-Libc**: Redirect execution to existing library code (e.g., `system("/bin/sh")`).<br><br>  <br><br>**ROP**: Chain existing code "gadgets".|
|**ASLR**|Randomizes memory address offsets for stack, heap, libraries, and binary text.|**Information Leak**: Disclosing addresses.<br><br>  <br><br>**Heap Spraying**: Allocating many copies of payload + NOP sleds into heap.|
|**Control Flow Integrity (CFI)**|Restricts indirect branches/calls/returns strictly to valid target paths defined in the Control Flow Graph (CFG).|Fine-grained CFG gadgets that do not violate the assigned label check.|

### Return-Oriented Programming (ROP)

- **Mechanism**: Bypasses $W \oplus X$ (DEP) by reusing existing executable code fragments ending in a `ret` instruction ("gadgets").
    
- **Execution**: Gadgets are chained together on the stack; executing `ret` pops the next gadget address off the stack into `%eip`.
    

## 4. Secure System Design & Memory Isolation

### A. Design Principles

- **Least Privilege**: Entities should operate with the bare minimum permissions necessary to complete their job.
    
- **Privilege Separation**: Divide a system into isolated components operating at different privilege levels.
    
- **Complete Mediation**: Every access attempt to every resource across a trust boundary must be validated against access policies.
    
- **Keep-It-Simple / Economy of Mechanism**: Keep the Trusted Computing Base (TCB) minimal and simple to reduce attack surfaces.
    
- **Defense-in-Depth**: Layer multiple independent safety mechanisms.
    

### B. Isolation Mechanisms

- **Process Isolation**: Ensures processes cannot directly inspect or modify another process's address space.
    
- **Virtual Memory**: Hardware (MMU) translates virtual addresses to physical addresses using multi-level **Page Tables**.
    
- **Hardware Privilege Rings**:
    
    - **Ring 0 (Kernel)**: Highest privilege level; executes core OS functions.
        
    - **Ring 3 (User Applications)**: Lowest privilege level; enforced entry points via system calls transition execution to Ring 0.
        

## 5. Side-Channel Attacks

- **Definition**: Exploits physical or operational implementation characteristics (timing, power, heat, cache behavior) rather than flaws in the abstract software design.
    

### Cache Side Channels

- Cache memory is a shared system resource across processes and execution levels.
    
- **Attack Strategies**:
    
    1. **Evict & Time**: Evict a target cache line, run victim code, and time the execution duration to infer access patterns.
        
    2. **Prime & Probe**: Fill cache lines with attacker data, let the victim execute, then measure access times across cache lines to identify evicted entries.
        
    3. **Flush & Reload**: Use memory-sharing (e.g., shared libraries) to flush specific cache lines via instructions like `clflush`, let the victim execute, then reload and time the access to determine access.
        

### Spectre & Meltdown

- **Meltdown**: Exploits out-of-order execution to bypass privilege isolation between Ring 3 (User) and Ring 0 (Kernel). Leaks kernel memory via cache side channels.
    
- **Spectre**: Exploits branch prediction / speculative execution to trick benign code into speculatively executing operations that leak secrets through cache timing.
    

Would you like to walk through a stack layout tracing problem or practice sample multiple-choice questions on any specific topic?