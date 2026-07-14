# Question 0
Complete the Nachos VM Worksheet included at the end of this document ([[#Nachos VM Worksheet|here]])

---
# Question 1
When using physical addresses directly, there is no virtual to physical translation overhead. Assume it takes **100 nanoseconds** to make a memory reference. If we used physical addresses directly, then all memory references will take 100 nanoseconds each.

* **a)** If we use virtual addresses with page tables to do the translation, then without a TLB we must first access the page table to get the appropriate page table entry (PTE) for translating an address, do the translation, and then make a memory reference. Assume it also takes **100 nanoseconds** to access the page table and do the translation. What is the effective memory reference time (time to access the page table + time to make the memory reference)?
**SOLUTION**
$$
\begin{align*}
&\text{time to access page table} + \text{time to make the memory reference}\\
&= 100ns + 100ns\\
&= \boxed{200ns}
\end{align*}
$$
* **b)** If we use a TLB, PTEs will be cached so that translation can happen as part of referencing memory. But TLBs are very limited in size and cannot hold all PTEs, so not all memory references will hit in the TLB. Assume translation using the TLB adds no extra time and the TLB hit rate is **75%**. What is the effective average memory reference time with this TLB?
**SOLUTION**

$$
\begin{align*}
\text{Average Memory Reference Time} &= 0.75 \times 100ns + 0.25 \times 200ns\\
&=\boxed{125ns}
\end{align*}
$$

* **c)** If we use a TLB that has a **99.5%** hit rate, what is the effective average memory reference time now? (This hit rate is close to what TLBs typically achieve in practice.)
**SOLUTION**
$$
\begin{align*}
\text{Average Memory Reference Time} &= 0.995 \times 100ns + 0.005 \times 200ns\\
&= 99.5ns + 1ns\\
&= \boxed{100.5ns}
\end{align*}
$$

---
# Question 2
Consider a 32-bit system (both virtual and physical addresses are 32 bits) with **1K pages** (pages of size 1024 bytes) and simple single-level paging.

* **a)** With 1K pages, the offset is 10 bits. How many bits are in the virtual page number (VPN)?
**SOLUTION**: 
$$
\begin{align*}
\text{bits in VPN} &= 32 - 10\\
&= \boxed{22bits}
\end{align*}
$$

* **b)** For a virtual address of `0xFFFF`, what is the virtual page number?
**SOLUTION**
$$
\begin{align*}
0xFFFF &= 0x00000000FFFFFFFF\\
&= 0000\ 0000\ 0000\ 0000\ 1111\ 1111\ 1111\ 1111_{2}\\
&\text{remove 10 bits}\\
&= 0000\ 0000\ 0000\ 0011\ 1111_{2}\\
&= 3F_{16} = \boxed{63}
\end{align*}
$$

* **c)** For a virtual address of `0xFFFF`, what is the value of the offset?
**SOLUTION**
$$
\begin{align*}
\text{offset} &= \text{last } 10 \text{ bits}\\
0xFFFF &= 0000\ 0000\ 0000\ 0000\ 1111\ 1111\ 1111\ 1111\\
&\text{10 bits remaining}\\
&= 11\ 1111\ 1111\\
&= 3FF_{16} = \boxed{1023}
\end{align*}
$$

* **d)** What is the physical address of the base of physical page number `0x4`?
**SOLUTION**
$$
\begin{align*}
ppn &= 4_{16} = 4_{10}\\
paddr &= ppn \times pageSize\\
&= 4 \times 1024\\
&= 4096 = \boxed{1000_{16}}
\end{align*}
$$
* **e)** If the virtual page for `0xFFFF` is mapped to physical page number `0x4`, what is the physical address corresponding to the virtual address `0xFFFF`?
**SOLUTION**
$$
\begin{align*}
\text{offset} &= 1023 = 0x3FF\\
\text{physical base address} &= 0x1000\\
paddr &= 0x1000 + 0x3FF = \boxed{0x13FF}
\end{align*}
$$

# Question 3
Suppose we have a computer system with a **44-bit virtual address**, page size of **64K**, and **4 bytes** per page table entry.

* **a)** How many pages are in a virtual address space? (Express using exponentiation.)
**SOLUTION**
$$
\begin{align*}
\text{offset size} &= \log_{2}(64 \times 1024) = \log_{2}(2^{6+10}) = 16\\
\text{virtual page number size} &= 2^{44 - 16} = \boxed{2^{28}}
\end{align*}
$$

* **b)** Every process has its own page table, and every page table has a root page and many secondary pages. Suppose we arrange for each kind of page table page (root and secondary) to have the size of a single page frame. How will the bits of the address be divided up among the offset, index into a secondary page, and index into the root page?
**SOLUTION**

$$
\begin{align*}
\text{offset} = \text{page size} &= 2^{16}\\
\text{remaining virutual address space} &= 44 - 16 = 28\\
\text{same size for root and secondary page} &\to \frac{28}{2} \\
= \text{bits for root and secondary page tables} &= \boxed{14} \\ \\ \\


\boxed{14\ bits: \text{root page table}} \boxed{14\ bits: \text{secondary page table}}&\boxed{16\ bits: \text{offset}}
\end{align*}
$$

* **c)** Suppose we have a **4 GB program** such that the entire program and all necessary page tables (using two-level pages from above) are in memory. How much memory, in page frames, is used by the program, including its page tables?
**SOLUTION**

$$
\begin{align*}
\text{Entries per page} &= \frac{2^{16}}{2^2} = 2^{14}\\
\text{Secondary page table memory} &= 2^{14} \times \frac{64K}{page} = 2^{14} \times 2^{16} = 2^{30}\\ \\

\text{Program frames} &= \frac{4GB}{64KB} = \frac{4 \times 2^{30}}{2^{16}} = 4 \times 2^{14}\\
\text{Total page table frames} &= 1 + 4 = 5\\ \\

\text{Total Memory} &= \text{Page table memories} + \text{Total page tables} \\
&= 2^{30} + 5 = \boxed{65541}
\end{align*}
$$

# Question 4
Suppose we have an average of one page fault every **20,000,000 instructions**, a normal instruction takes **2 nanoseconds**, and a page fault causes the instruction to take an additional **10 milliseconds**.

* What is the average instruction time, taking page faults into account?
**SOLUTION**

$$
\begin{align*}
p &= \frac{1}{20000000}\\
\text{Average instruction time} &= p \times (2ns + 10ms) + (1-p) \times 2ns\\
&= p \times (2 + 10000000)ns + (1-p) \times 2ns\\
&= 10000002p + 2 - 2p\\
&= 10000000p + 2\\
&= 0.5 + 2\\
&= \boxed{2.5ns}
\end{align*}
$$
* Redo the calculation assuming that a normal instruction takes **1 nanosecond** instead of 2 nanoseconds.
**SOLUTION**
$$
\begin{align*}
\text{Average Instruction Time} &= p \times (1ns + 10ms) + (1-p) \times 1ns\\
&= 10000001p + 1 - p\\
&= 10000000p + 1\\
&= 0.5 + 1\\
&= \boxed{1.5ns}
\end{align*}
$$

# Question 5
If **FIFO** page replacement is used with four page frames and eight pages (numbered 0-7), how many page faults will occur with the reference pattern `427253323126` if the four frames are initially empty? Which pages are in memory at the end of the references? Repeat this problem for **LRU**.

**SOLUTION**
1. **FIFO**
```
4 2 7 2 5 3 3 2 3 1 2 6  (page numbers)
^ ^ ^   ^ ^       ^ ^ ^  (page fault)
/ / /                    (removed from frames for space)
```

$$
\text{number of page faults} = \text{number of } \textasciicircum \text{ arrows} = \boxed{8}

$$
2. **LRU**
```
4 2 7 2 5 3 3 2 3 1 2 6  (page numbers)
^ ^ ^   ^ ^       ^   ^  (page fault)
/   /   /                (removed from frames for space)
```

$$
\text{number of page faults} = \text{number of } \textasciicircum \text{ arrows} = \boxed{7}
$$
# Question 6
For each of the following actions, explain whether it will trigger a fault or not. If it will trigger a fault, briefly describe what kind of fault will occur (e.g., page fault, protection fault) and how the OS will handle it.

* **a)** A process accesses memory in a page that is currently swapped out to disk.
**SOLUTION**
Page fault since process cannot find the memory page in the current TLB.
OS will get it back from disk to TLB and redo the process again.

* **b)** A process tries to dereference a NULL pointer (i.e., access memory at address `0x00`).
**SOLUTION**
Protection fault since dereferencing a NULL pointer is considered illegal action.
OS terminates to process and flags the process of segmentation fault occurred.

* **c)** A process calls `malloc`.
**SOLUTION** 
No faults since `malloc` is a standard library to allocate heap memory.

* **d)** A process calls a function, which pushes new variables onto the stack, extending beyond the current bounds of the stack.
**SOLUTION**
Page fault since no space in current pages to store the new variable
OS grabs a new page frame for the compensation of the growth of the stack

* **e)** A child process reads from a page that is shared with its parent using copy on write.
**SOLUTION**
No faults since child will get the value of  the data from the parent's page into its own during copy on write

* **f)** A child process writes to a page that is currently set to read-only due to copy on write.
**SOLUTION**
Protection and Page fault since illegal to write in read-only page and child don't have that page's ownership
OS allocate new page for child, copy the content from the parent's page, and set this new page to read-write

# Question 7
Consider a demand-paging system with the following utilizations:
* CPU utilization: **20%**
* Paging disk: **97.7%** (demand, not storage)
* Other I/O devices: **5%**

For each of the following, say whether it will (or is likely to) improve CPU utilization. Briefly explain your answers.

* **a)** Install a faster CPU.
**SOLUTION**
No, a faster CPU means a faster process on CPU related tasks but does not utilize the CPU more

* **b)** Install a bigger paging disk.
**SOLUTION**
No, a bigger paging disk helps hold more data in disk, which does not help demand for paging disks, therefore does not help with CPU utilization

* **c)** Increase the degree of multiprogramming.
**SOLUTION**
No, this actually adds more programs to run at the same time, which uses already limiting amount of physical pages and potentially adding more paging requests

* **d)** Decrease the degree of multiprogramming.
**SOLUTION**
Yes, this decreases the load on the physical memory, allowing less demand to page the disk and therefore allowing CPU to do more work

* **e)** Install more main memory.
**SOLUTION**
Yes, by adding more main memory helps decrease the probability of having to page the disk from having no current page in physical memory that does not hold the memory needed for the current process, allowing less time spent on swapping memory and more time on CPU doing work

* **f)** Install a faster hard disk, or multiple controllers with multiple hard disks.
**SOLUTION**
Yes, by speeding up the process of swapping pages from disk, it decreases the amount of time spent in I/O and more time in CPU

* **g)** Add prefetching to the page replacement algorithm.
**SOLUTION**
No, due to the high paging demand, this will spend more time in disk and makes it utilize less time on CPU on speculative pages that might end up not utilizing. 

* **h)** Increase the page size.
**SOLUTION**
No, this makes it worse when the paging was demanded since a bigger page size means a longer time to actually swap the pages from disks.

---

# Nachos VM Worksheet

In part 2 of project 2, you will be creating and initializing the page tables used by user-level processes running on Nachos. This worksheet is intended to give you practice with Nachos page tables so that you are comfortable understanding how virtual memory works in Nachos. The page table for a user-level process is represented by the `TranslationEntry[] pageTable` array in the `UserProcess` class. In the `UserProcess` constructor, you will be initializing each `TranslationEntry` in the array to have each virtual page point to an allocated physical page.

**Assumptions for this worksheet:**
* `Processor.PageSize` = 128 bytes (`0x80`)
* Program requires **4 pages**
* Physical memory has **8 pages**

![[Pasted image 20260510172932.png]]

### Exercises

**Question 1:** What is the value of `UserProcess.numPages`?
`UserProcess.numPages = 4`

**Question 2:** What is the value of `Processor.numPhysPages`?
`Processor.numPhysPages = 8`

**Question 3:** Fill in the values of the `UserProcess.pageTable` mappings:
* `pageTable[0].vpn = 0`, `pageTable[0].ppn = 2`
* `pageTable[1].vpn = 1`, `pageTable[1].ppn = 0`
* `pageTable[2].vpn = 2` , `pageTable[2].ppn = 6`
* `pageTable[3].vpn = 3`, `pageTable[3].ppn = 4`

**Question 4:** What is the virtual address of virtual page 2?
*(Hint: virtual address = vpn × PageSize)*
$$
\text{virtual address } = 2 \times \text{pageSize} = 256_{10} = 100_{16}
$$

**Question 5:** What is the physical address of virtual page 2?
*(Hint: physical address = ppn × PageSize)*
$$
\text{physical address} = 6 \times \text{pageSize} = 768_{10} = 300_{16}
$$

**Question 6:** What physical page does the virtual address **298** reside in?
*(Hint: vpn = virtual address / PageSize (integer division), then look up the corresponding ppn)*
$$
\begin{align*}
\text{vpn} = \frac{298}{128} &= 2\\
\text{pageTable[2].ppn} &= 6\\
298 \text{ is in page table } &\boxed{6}
\end{align*}
$$

**Question 7:** What is the offset of the virtual address **298**?
*(Hint: offset = virtual address mod PageSize)*

$$
\text{offset} = 298 \ \%\  128 = \boxed{42}
$$

**Question 8:** What is the physical address of the virtual address **298**?
*(Hint: physical address = ppn × PageSize + offset)*
$$
\text{physical address}=6 \times 128 + 42 = 810_{10} = 32A_{16}
$$

**Question 9:** Why does the simple `System.arraycopy` no longer work for `readVirtualMemory` and `writeVirtualMemory` in the general case?
*(Hint: Consider the case when vaddr is 0x7E and data.length is 4 with the page table above.)*

Because there could be a memory address where the data spans across 2 page tables which the data is scattered in 2 separate locations. The general `System.arraycopy` assumes continuous memory, but it is not guaranteed to be the case in physical memory.