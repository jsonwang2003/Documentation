### 1. The Virtual Memory Illusion
When a process runs, it does not interact with the physical RAM sticks directly. Instead, the Operating System provides each process with its own private **Virtual Address Space**.
- **The Illusion**: Every process thinks it has the entire memory to itself, starting at address `0x00...0` up to a maximum limit.
- **The Reality**: The OS and the CPU (via the Memory Management Unit, or **MMU**) map these virtual addresses to actual **Physical Addresses** in hardware.

---
### 2. Why the Parent and Child share the "Same" Address
When `fork()` is called:
1. The OS creates a **new address space** for the child.
2. It copies the parent's page tables (the "map" of memory) to the child.
3. Because the map is a copy, the variable `vals` exists at the **exact same virtual address** (e.g., `0x7ffee...`) in both processes.

---
### 3. Copy-on-Write (CoW)
To save time and memory, the OS doesn't actually copy the data of the array immediately. It marks the memory as **Read-Only** for both.
- When the child tries to modify the data (`vals[0] = 9999`), a "page fault" occurs.
- The OS steps in, realizes this is a forked process, and **creates a unique physical copy** of that memory page for the child.
- The child's virtual address remains the same, but it now points to a different physical location than the parent.

---
### 4. Analyzing the Possibilities

|**Possibility**|**Result**|**Why?**|
|---|---|---|
|**Address**|**Same**|`fork()` copies the address space layout exactly.|
|**Value**|**Different**|Processes are **isolated**. Changing memory in the child does not affect the parent.|

This isolation is why the parent prints `1` even after the child sets the value to `9999`. They are looking at the same "house number" (Virtual Address) but on two completely different "streets" (Physical RAM).

---
### 5. Benefits of Virtual Memory
- **Isolation**: A bug or crash in the child process cannot corrupt the parent's memory.
- **Security**: One process cannot "peek" into the memory of another process because it simply doesn't have a map to that physical space.
- **Efficiency**: The OS can load only the parts of a program currently being used (Demand Paging).