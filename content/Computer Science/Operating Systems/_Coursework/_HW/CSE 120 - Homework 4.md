Spring 2026 
Due: Tuesday, June 2nd at 11:59pm 

---
## Question 1 
On a Unix-style file system, how many disk read operations are required to open and read the first block of the file "`/usr/include/X11/Xlib.h`"? Assume: 
- The superblock is in memory, but nothing else
- All directories and inodes are one block in size
- Once a block is read, it is cached in memory (and does not need to be read again)
- inode timestamps (like last access) do not need to be updated (the inode does not need to be written after access). 

**SOLUTION**
1. Read superblock disk for root (`/`) inode
2. Read `/` directory for `user` inode
 3. Read `usr` inode
 4. Read `usr` directory for `include` inode
 5. Read `include` inode
 6. Read `include` directory for `X11` inode
 7. Read `X11` inode
 8. Read `X11` directory for `Xlib.h` inode
 9. Read `XLib.h` inode
 10. Read `Xlib.h` inode number for `XLib.h` block

Result: total of $10$ reads from disk to read the first block of  "`/usr/include/X11/XLib.h`"

---
## Question 2 
Consider a UNIX-style inode with 10 direct pointers, one single-indirect pointer, and one double-indirect pointer only. Assume that the block size is 4KB (including indirect blocks), and that the size of a pointer is 4 bytes. How large a file can be indexed using such an inode? 

**SOLUTION**
$$
\begin{align*}
\text{Number of pointers in an indirect block} &= \frac{4KB}{4B} = 1024\\ \\
\text{Size with direct pts} &= 4KB \times 10 = 40KB\\
\text{Size with single-indirect block} &= 1024 \times 4KB = 4096KB\\
\text{Size with double-indirect block} &= 1024 \times 1024 \times 4KB = 4194304KB\\ \\
\text{total} &= 4194304 + 4096 + 40 = \boxed{4198440KB}
\end{align*}
$$
---
## Question 3 
The original Berkeley Fast File System increased the Unix file system block size from 512 bytes to 4096 bytes. Concerned about internal fragmentation, the FFS also introduced the ability to end a file with a small fragment. A disk block could be broken up into small fixed-size fragments, each of which could be used to store the ends of different files. For instance, a file of size 5000 bytes would need two blocks (8192 bytes) to store on disk, resulting in 3192 bytes lost to internal fragmentation. With 1024-byte fragments, though, the file could be stored with one full-sized block (4096 bytes) and one 1024-byte fragment, requiring 5120 bytes on disk to store the file and reducing internal fragmentation to just 120 bytes. The tradeoff is that managing fragments increases the complexity of the file system implementation. 

My laptop has $2^{20}$ $(1024 \times 1024)$ files on it. Assume the disk block size is 4KB and the average amount of internal fragmentation is 2KB per file.

### (a) How much storage is wasted due to internal fragmentation in the file system on my laptop?

**SOLUTION**
$$
\begin{align*}
&\text{Average internal fragmentation} \\
&= 2^{20} files \times 2 \frac{KB}{files} \\
&= 2^{21} KB \\
&= \boxed{2GB}
\end{align*}
$$

### (b) Assume that, with fragments, the average amount of internal fragmentation goes down to 256 bytes per file. How much storage is wasted due to internal fragmentation when using fragments?

**SOLUTION**
$$
\begin{align*}
&\text{Average internal fragmentation after fragments} \\&= 2^{20} files \times 256 \frac{B}{files} \\
&= 2^{20} files \times 2^{8} \frac{B}{files} \\
&= 2^{28} B \\
&= \boxed{256 MB}
\end{align*}
$$
### (c) Assume that you would receive the same benefits for your laptop. Would you want the file system to use fragments to save space? Why or why not? 

**SOLUTION**
As the problem statement suggested, this benefit comes at the tradeoff of the file system being more complicated. Due to such tradeoff, the overhead for accessing files is most likely having a larger overhead, effective slowing and process to the file. Therefore I would not want the file system to use fragments to save space as it damages the efficiency.


---
## Question 4 
Consider a file archival system, like the programs `zip` or `tar`. Such systems copy files into a backup file and restore files from the backup. For example, from the `zip` documentation:

> The zip program puts one or more compressed files into a single `zip` archive, along with information about the files (name, path, date, time of last modification, protection, and checksum information to verify file integrity). 

When a file is restored, it is given the same name, time of last modification, protection, and so on. If desired, it can even be put into the same directory in which it was originally located. 

Can `zip` restore the file into the same inode as well? Briefly explain your answer. 

**SOLUTION**
No, `zip` can only open a new file's inode and assign it to some disk's location, it cannot guarantee that the OS gives the same inode again when `zip` restores the file

---
## Question 5 
How does a file cache help improve performance? Why do systems not use much larger caches if they are so useful? 

**SOLUTION**
A file cache helps with performance by keeping copies of recently/frequently accessed file data in RAM instead of going to disk storage every time. 

Systems no use larger caches for the following reasons:
- **Cost**: more expansive to store more things into RAM
- **Shared Space**: file cache share the same physical space in RAM, the bigger the file cache, the smaller the user process memory has, causing starvation for user process

---
## Question 6 
Consider a program that executes a loop that issues a read I/O to a storage device and waits I milliseconds for the I/O to complete, and then computes on the data returned for X milliseconds, and then repeats. 

For various values of I and X, compute the percentage of time that the program spends waiting for I/O and fill in the following table. If I and X are both 1 ms, for example, then the program spends 50% of its time waiting for I/O. 


|                             | X = 100ms | X = 10ms | X = 1ms | X = 0.1ms |
| --------------------------- | --------- | -------- | ------- | --------- |
| I = 25ms<br>(cloud storage) | 20%       | 71.4%    | 96.2%   | 99.6%     |
| I = 5ms<br>(HDD)            | 4.8%      | 33.3%    | 83.3%   | 98%       |
| I = 0.1ms<br>(SSD)          | 0.099%    | 0.909%   | 9.09%   | 50%       |
| I = 0.0005ms<br>(NVM)       | 0.0005%   | 0.005%   | 0.05%   | 0.5%      |
| I = 0.0001ms<br>(RAM)       | 0.0001%   | 0.001%   | 0.01%   | 0.1%      |

---
## Question 7 
Consider a process running in a virtual machine. Assume that the virtual machine is running on a type 1 hypervisor with no modern hardware support for virtualization. For each of the following events, which system component(s) are involved in handling it? For each answer, choose from: the guest OS, the hypervisor, both, or neither. Briefly explain each answer. 
### (a) The process writes to a file. 

**SOLUTION**
Both

the process calls a syscall to handle write from the *guest OS*. The guest OS processes this syscall and tries to issue hardware I/O. But since there is no support for virtualization, the *type 1 hypervisor* traps the privileged instruction, runs the instruction themselves and return the control back to user
### (b) The process calls exit(). 

**SOLUTION**
Guest OS

`exit()` syscall only terminates the current process and return resources such as memory and file descriptors. These can be handled entirely through *guest OS* and does not require any privileged instructions for the syscall `exit()`.
### (c) A page fault occurs on a page that the hypervisor has swapped out to disk.

**SOLUTION**
Hypervisor

Since the *hypervisor* is responsible for managing the physical memory. If a page was swapped out to disk, it can only be swapped back by hypervisor
### (d) A context switch between two user-level threads (e.g., as a result of yield()). 

**SOLUTION**
Neither

Since a context switch between 2 threads are in user-level, it never switches from user mode to kernel mode, therefore it will not be processed by guest OS. And no privileged instruction so no hypervisor
### (e) The process divides by zero. 

**SOLUTION**
Both

Since this error is from the hardware, this must pass through the hypervisor. After which can recognize that the error is from some VM running on the machine and redirect the exception to the guest OS
### (f) A timer interrupt triggers a switch from running this virtual machine to running a different virtual machine on the same computer. 

**SOLUTION**
Hypervisor

the timer is a hardware component, which catches the time interrupt by a hypervisor. the hypervisor switches from one VM to another by pausing the current one, save its states, and restores the state to the second VM. Throughout this process, guest OS is totally unaware of this.

---
## Question 8 
Consider a system that supports $5000$ users. Suppose that you want to allow $4990$ of these users to be able to access one file. 

### (a) How would you specify this protection scheme in Unix?

**SOLUTION**
1. Create a group of 10 users who should not have access to the file
2. Change group ownership by setting the file's group ownership to this created group of 10 users
3. Set the file permissions so that the group has zero access and others have the access
### (b) Could you suggest another protection scheme that can be used more conveniently for this purpose than the scheme provided by Unix?

**SOLUTION**
For the file, have general access to any user except for a list of 10 specific users kept by the file's metadata