Due: 

Checkpoint: Wednesday May 27th (11:59pm)

Final: Friday June 5th (11:59pm)

The last project! In project 2, each process had a page table that was initialized with physical pages and their contents when the process was created. In project 3, you will implement a more sophisticated memory management system where physical pages are allocated on demand and pages that cannot fit in physical memory will be stored on disk. 

## Background
You will implement and debug virtual memory in two main steps. First, you will implement demand paging using page faults to dynamically initialize process virtual pages on demand, rather than initializing page frames for each process in advance at exec time as you did in project 2. Next, you will implement page replacement, enabling your kernel to evict a virtual page from memory to free up a physical page frame to satisfy a page fault. Demand paging and page replacement together allow your kernel to "overbook" memory by executing more processes than would fit in machine memory at any one time, using page faults to multiplex the available physical page frames among the larger number of process virtual pages. When implemented correctly, virtual memory is undetectable to user programs unless they monitor their own performance. 

You project will implement the following functionality:

- **Demand Paging.** Pages will be in physical memory only as needed. When no physical pages are free, it is necessary to free pages, possibly evicting pages to swap.
- **Lazy Loading.** To fulfill the spirit of demand paging, processes should load no pages when started, and depend on demand paging to provide even the first instruction they execute. When you are done, loadSections will not allocate even a single page.
- **Page Pinning.** At times it will be necessary to "pin" a page in memory, making it temporarily impossible to evict.

The changes you make to Nachos will be in these two files in the vm directory:

- `VMKernel.java` — an extension of UserKernel  
- `VMProcess.java` — an extension of UserProcess

These classes inherit from UserKernel and UserProcess. While the VM versions of these classes will be able to depend upon functionality in the base classes, the focus in this project will be on demand-paged virtual memory. As a result, you will be implementing new versions of key methods in VMProcess such as loadSections, readVirtualMemory, and writeVirtualMemory.

You will compile and run the project in the proj3 directory. Unlike the first two projects, you will not need to learn any new Nachos modules and will continue to use functionality that you became familiar with in project 2. Before starting your implementation, also see the Tips section below.

## Design Aspects
Central to this project are the following design aspects:

### TranslationEntry bits
You will extend your kernel's handling of the page tables to use three special bits in each TranslationEntry (TE):

- **Valid bit:** You will set or clear the valid bit in each TE to tell the CPU which virtual pages are resident in memory (a valid translation) and which are not resident (an invalid translation). If a user process references an address for which the TE is marked invalid, then the CPU raises a page fault exception and transfers control to the Nachos exception handler. You will use the valid bit starting with part 1.

- **Used bit:** The CPU sets the used bit (aka reference bit) in the TE to pass information to Nachos about page access patterns. If a virtual page is referenced by a process, the machine sets the corresponding TE used bit to inform the kernel that the page is active. Once set, the used bit remains set until the kernel clears it. You will not use the used bit until part 2.

- **Dirty bit:** The CPU sets the dirty bit in the TE whenever a process executes a store (write) to the corresponding virtual page. This step informs the kernel that the page is dirty; if the kernel evicts the page from memory, then it must first "clean" the page by writing its contents to disk. Once set, the dirty bit remains set until the kernel clears it. You will not need to use the dirty bit until part 3.

### Swap File
To manage swapped out pages on disk, use the StubFileSystem (via ThreadedKernel.fileSystem) as in project 2. There are many design choices, but we suggest using a single, global swap file across all processes. This file should last the lifetime of your kernel. Be sure to choose a reasonably unique file name that will not conflict with other files in the test directory. When designing the swap file, keep in mind that the units of swap are pages. Thus you should be efficient with disk space using the same techniques applied in virtual memory: any gaps in your swap space due to processes terminating should be used by future processes. As with physical memory in project 2, a global free list works well. You can assume that the swap file can grow arbitrarily, and that there should not be any read/write errors. Assert if there are.

### Global Memory Accounting
In addition to tracking free pages (which may be managed as in project 2), there are now two additional pieces of memory information relevant to all processes: which processes own which pages (part 2), and which pages are pinned (part 4). The former is necessary to manage eviction of pages, and the latter is necessary when optimizing page fault handling to use the virtual memory subsystem in a more fine-grained manner. There are also multiple approaches to solving this problem, but we suggest using a global inverted page table (see the tips below).

## Tasks

### Checkpoint [7%]
Implement the first step of true demand paged virtual memory: lazy physical page allocation without page replacement. This checkpoint is a subset of part 2 below and is intended to be an incremental part of your final project 3 implementation. For the final project, you will extend this implementation by adding page replacement, swap, dirty bit optimization, and page pinning.

For the checkpoint, change VMProcess.loadSections so that it does not allocate a physical page for every virtual page at exec time. Instead, initialize the page table entries as invalid, and allocate a physical page only when the process faults on that virtual page. On a page fault, allocate one free physical page, initialize it with the correct contents, update the faulting TranslationEntry, mark it valid, and return from the exception without advancing the PC. See task 2 for the full version of this functionality.

The checkpoint does not require page replacement, swap, the clock algorithm, dirty bit optimization, or page pinning. If there are no free physical pages, your checkpoint implementation may fail; handling that case is part of the final project. The checkpoint tests will focus on programs whose total virtual address spaces may be larger than physical memory, but whose actually touched pages fit in physical memory.

The checkpoint is the intended first step toward Part 2. If you complete the checkpoint, do not undo that design later: VMProcess.loadSections should not allocate physical pages, and physical pages should be allocated on demand during page faults.

### Part 1 [30%]
Implement demand paging. In this part, you will implement the basic page fault handling needed for demand paging, assuming that a free physical page frame is available when a fault occurs. If you completed the checkpoint, keep that design: do not allocate physical page frames for every virtual page at exec time. Instead, create invalid `TranslationEntries` in `loadSections` and allocate a physical page frame when the process first faults on that virtual page. If no free physical page frame is available, for now it is acceptable to fail; handling that case with page replacement and swap is part 2. The only special bit in `TranslationEntries` that you need to use for this part is the valid bit. You will not yet need to implement the swap file, page replacement, an inverted page table, etc. Instead, you just need to make the following changes:

- In `VMProcess.loadSections`, initialize all of the `TranslationEntries` as invalid. This will cause the machine to trigger a page fault exception when the process accesses a page. Also do not allocate a physical page or initialize the page by, e.g., loading from the COFF file. Instead, you will allocate and initialize the page on demand when the process causes a page fault. Note that handling a page fault does not have a return value.

- Handle page fault exceptions via `VMProcess.handleException`. When the process references an invalid page, the machine will raise a page fault exception (if a page is marked valid, no fault is generated). Modify your exception handler to catch this exception and handle it by preparing the requested page on demand. The Processor class lists all of the exceptions and associated registers that the MIPS CPU can generate, and it has one of each for page faults.

- Add a method to prepare the requested page on demand. Note that faults on different pages are handled in different ways. A fault on a page in the COFF should read the corresponding code page from the COFF file, and a fault on a stack page or arguments page should zero-fill the page (set every byte on the page to 0). For this step, for reference look at the COFF file loading code from `UserProcess.loadSections` from project 2. If the process faults on page 0, for example, then load the first page of code from the executable file into it. More generally, when you handle a page fault you will use the value of the faulting address to determine how to initialize that page: if the faulting address is in a segment of the COFF file, then load the appropriate page; if it is any other page, zero-fill it. It is fine to loop through the sections of the COFF file until you find the appropriate section and page to use (assuming it is in the COFF file).

- Once you have paged in the faulted page, mark the `TranslationEntry` as valid. Then let the machine restart execution of the user program at the faulting instruction: return from the exception, but do not increment the PC (as is done when handling a system call) so that the machine will re-execute the faulting instruction. If you set up the page (by initializing it) and page table (by setting the valid bit) correctly, then the instruction will execute correctly and the process will continue on its way, none the wiser.

At this point we recommend doing task 2 for test programs that do not use files or the console, such as `matmult`, swap4 and swap5 (see "where and how to focus time" in the general tips). Then come back and implement new `VMProcess.readVirtualMemory` and `VMProcess.writeVirtualMemory` methods to handle invalid pages and page faults. Start with your implementations from project 2, or create new ones, by implementing the methods for `VMProcess`. Both methods directly access physical memory to read/write data between user-level virtual address spaces and the Nachos kernel. These methods will now need to check to see if the virtual page is valid. If it is valid, it can use the physical page as before. If the page is not valid, then it will need to use the page fault handler to bring the page in as with any other page fault.

### Testing
As long as there is enough physical memory for all pages the program actually touches, then you should be able to use test programs from project 2 to test this part of project 3. See the tips in the Testing section below for how you can control (increase or decrease) the number of physical pages (e.g., write10 is going to need more than the default of 16 pages). If you give Nachos enough physical pages, you can even run the swap4 and swap5 tests (and these tests do not use any system calls other than exit).

### Part 2 [51%]
Now implement demand paged virtual memory with page replacement. In this second part, not only do you delay initializing pages, but now you delay the allocation of physical page frames until a process actually references a virtual page that is not already loaded in memory.

If your implementation still `preallocates` physical frames in `VMProcess.loadSections`, change it so that it does not allocate a physical page for every virtual page. If you completed the checkpoint, this should already be done. All `TranslationEntries` should start invalid, and physical frames should be allocated on demand during page faults.

Extend your page fault exception handler to handle the case where no free physical page frame is available. If a free frame exists, your checkpoint/part 1 code should already allocate it, initialize the page, mark the `TranslationEntry` as valid, and return from the exception. In this part, add page replacement and swap so that page faults can still be handled when physical memory is full.

You can get the above two changes working without having page replacement implemented for the case where you run a single program that does not consume all of physical memory. Before moving on be sure that the two changes above work for a single program that fits into memory.

Now implement page replacement to free up a physical page frame to handle page faults:

- Extend your page fault exception handler to evict pages once physical memory becomes full. First, you will need to select a victim page to evict from memory. Your page eviction strategy should be the clock algorithm as described in lecture. For this part, use the used bit in `TranslationEntries` to track when pages are used by a process.

- Evict the victim page to the swap file and mark the `TranslationEntry` for that page as invalid.

- Read in the contents of the faulted page (more below).

- Implement the swap file for storing pages evicted from physical memory. You will want to implement methods to create a swap file, write pages from memory to swap (for page out), read from swap to memory (for page in), etc.

- Implement an inverted page table (see the tips below) to keep track of which virtual pages are using which physical pages.

First implement this paging functionality for a single process. Once that works, extend your implementation to support multiple processes:

- For part 2 use a simple locking strategy to protect your data structures used for demand paging: acquire a lock when you start handling a page fault, and release it when you are done handling the page fault. (Part 4 will optimize the use of this lock further)

- After getting paging with multiple processes working with the memory tests and you then move on to `readVirtualMemory` and `writeVirtualMemory`, you should use the same lock to protect each page accessed in their while loops (e.g., so that between the time `rVM` brings a page into memory and you use `arrayCopy` on that page, the page is not chosen by another process for eviction).

- Modify your implementation for cleaning up a process when it terminates by (1) only freeing physical pages that have been allocated to it and (2) freeing all swap pages allocated by the process.    

Optional: We will not be testing the arguments to exec or any of the join functionality. However, if you have that functionality from project 2 and you would like to have it work for project 3, note that your implementation will need to check the validity of the arguments page (for exec) and the validity of the page with the status argument to join.

As you implement the above operations, keep the following points in mind:

- As in the first part of the project, the first time a page is touched it needs to be initialized. If this page is subsequently evicted to swap, it will be read from there on further page faults. To be concrete, consider a page used for data. On the first access (fault) on that page, you will read that page in from the executable file. When this page gets evicted, you should write it to the swap file. If the process faults on the page again, you should read the page in from the swap.

- Important: When supporting multiple processes, the page replacement algorithm may select a victim page to evict from another process. As a result, you will need to update the `TranslationEntry` in the page table for that process, not the faulting one.

### Testing
Start with the memory focused tests such as swap4 and swap5, and vary the amount of physical memory in Nachos to control how much demand paging each test must do. See the Testing section below for how to control the number of pages in physical memory.

### Part 3 [5%]
In part 2, when evicting pages we ignored the dirty bit, i.e., we ignored whether or not the process had modified the page once it was brought into memory. As a result, when evicting a page we always wrote it to the swap file. In this part you will optimize page replacement by using the dirty bit. Consider a page P that has been written to the swap file and then read back into physical memory on a page fault. If P is again chosen for eviction, your implementation will only write P back to the swap file if P has been modified (i.e., if the dirty bit is true in its `TranslationEntry`). If P is not dirty, then it does not need to be written to swap since the version of P in memory is the same as the one in the swap file. Note that (1) this also means that you should not free up a page in the swap file for a given virtual page until the process terminates, and (2) you will need to modify `writeVirtualMemory` to set the dirty bit on the page being written to (since these writes are not being done by the emulated CPU, the CPU does not set the dirty bit on these writes as it does on all other writes).

After implementing this optimization, you should only do as many page reads and writes to the swap file as necessary to execute the program, and as dictated by the page replacement algorithm.

Note that parts 3 and 4 do not depend on each other. You can implement part 3 before part 4, and vice versa.

### Testing
Use the same tests as with the previous parts. With this optimization, you should notice that the number of writes to the swap file (pageouts) decreases (sometimes significantly). In particular, you should notice a substantial difference in the number of pageouts between swap4 and swap5 since swap5 modifies the data in its memory.

### Part 4 [7%]
In part 2, when handling a page fault we recommended using a simple locking strategy that acquires and holds a lock for the duration of your page fault handler. While simpler to implement, this strategy limits performance: when one process is handling a page fault - which may take milliseconds to complete - no other process can use the virtual memory subsystem (create new processes, handle their own page faults, etc.). In this part, you will optimize page replacement by allowing multiple processes to use the virtual memory subsystem in a more fine-grained manner. Note that this optimization is only useful once you have paging support for multiple processes working.

There are two aspects to implementing this optimization. The first is that you will need to modify the locking strategy from part 2: the process must release the lock before performing an I/O operation to the COFF file or swap file, and acquire it again when the I/O operation completes. Put another way, the requirement is that while a process performs an I/O operation to COFF or swap, the process cannot hold any locks.

The second is that when your code is using a physical page for copying data with `readVirtualMemory` or `writeVirtualMemory`, or when it is performing an I/O operation to the COFF file or the swap file, it will need to "pin" the physical page while using it so that no other process