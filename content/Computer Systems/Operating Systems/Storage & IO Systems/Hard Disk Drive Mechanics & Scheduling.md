---
description: "Physical architecture of Hard Disk Drives (HDDs), mechanical latency calculations, the OS block interface abstraction, and disk scheduling algorithms."
aliases:
  - Hard Disk Drives
  - HDD
  - Disk Scheduling
  - Seek Time
  - Rotational Latency
  - SSTF
  - SCAN
  - Block Interface
tags:
  - operating-systems
  - storage
  - hdd
  - disk-scheduling
---
> [!abstract] Abstract
> **Hard Disk Drives (HDDs)** have served as the primary persistent storage mechanism for decades. Because HDDs rely on mechanical moving parts (spinning platters and moving actuator arms), disk access carries massive latency compared to physical RAM. The operating system abstracts complex physical disk geometries into a simple array of logical blocks and applies **Disk Scheduling Policies** to minimize physical arm movement and rotational delay.
> 
> - **Category:** Physical Storage & Mechanical Devices
> - **Primary Abstraction:** Logical Block Addressing (LBA / Block Interface).
> - **Bottleneck:** Physical seek time and rotational latency.

---

## Memory Hierarchy & Persistence

Storage devices occupy the lower tiers of the system memory hierarchy, offering vast non-volatile storage capacity at the cost of significantly higher latency:

![[Pasted image 20260727223932.png]]

```mermaid
graph TD
    CPU["CPU Registers (~1 cycle)"] --> Cache["L1/L2/L3 Hardware Caches"]
    Cache --> RAM["DRAM Main Memory (~100 ns)"]
    RAM --> SSD["Flash SSD (~10-100 µs)"]
    SSD --> HDD["Hard Disk Drive HDD (~1-10 ms)"]
```

While [[Virtual Memory & Address Translation Fundamentals|DRAM]] loses its state when power is removed, persistent storage devices maintain data indefinitely external to the CPU.

---

## HDD Physical Architecture

Hard Disk Drives read and write magnetic data using mechanical components operating in tight physical synchronization:

![[Pasted image 20260727224426.png]]

![[Pasted image 20260727224523.png]]

*   **Platters:** Circular magnetic disks mounted on a central rotating **Spindle**.
*   **Read/Write Head:** Electromagnetic sensors positioned nanometers above platter surfaces.
*   **Actuator Arm:** Rotates the read/write heads radially across concentric tracks on the platters.
*   **Cylinder:** The set of matching tracks aligned vertically across all platters.

### Reading Disk Sector
![[Pasted image 20260727225103.png|757]]
![[Pasted image 20260727225019.png]]

---

## Mechanical Access Latency Calculations

Total disk latency to read or write a sector is governed by three distinct physical phases:

$$\text{Disk Latency} = T_{\text{seek}} + T_{\text{rotation}} + T_{\text{transfer}}$$

```mermaid
flowchart LR
    A["1 Seek Phase<br/>Move actuator arm to target cylinder"] --> B["2 Rotation Phase<br/>Wait for sector to spin under head"]
    B --> C["3 Transfer Phase<br/>Read/write magnetic bits off platter"]
```

1.  **Seek Time ($T_{\text{seek}}$):** The mechanical delay required to swing the actuator arm to the correct track. This is bounded by physics and represents the slowest component.
2.  **Rotational Delay ($T_{\text{rotation}}$):** The time spent waiting for the target magnetic sector to rotate under the read/write head. Average rotational latency equals the time for half a revolution.
3.  **Transfer Time ($T_{\text{transfer}}$):** The time required to read data from the platter surface into the disk controller electronics.

### Sample Latency Calculation
Assume a disk with $15,000\text{ RPM}$, an average seek time of $4\text{ ms}$, and a transfer rate of $125\text{ MB/s}$ reading two $512\text{-byte}$ sectors ($1024\text{ bytes}$ total):

*   **Seek Time:** $T_{\text{seek}} = 4\text{ ms}$
*   **Average Rotational Latency:**
    $$T_{\text{rotation}} = \frac{1}{2} \times \left( \frac{1\text{ min}}{15000\text{ rev}} \times \frac{60000\text{ ms}}{1\text{ min}} \right) = \frac{1}{2} \times 4\text{ ms} = 2\text{ ms}$$
*   **Transfer Time:**
    $$T_{\text{transfer}} = \frac{1024\text{ Bytes}}{125 \times 10^6\text{ Bytes/sec}} = 8.192 \ \mu\text{s} \approx 0.008\text{ ms}$$
*   **Total Latency:**
    $$\text{Total Latency} \approx 4\text{ ms} + 2\text{ ms} + 0.008\text{ ms} = \mathbf{6.008\text{ ms}}$$

---

## The OS Block Interface Abstraction

Historically, operating systems specified raw disk accesses via **CHS** geometry (Cylinder, Head, Sector). Modern drives encapsulate this complexity inside internal disk controllers, exposing a simple **Logical Block Interface**:

```mermaid
graph LR
    OS["Operating System"] -->|"Read / Write Block N"| LBA["Logical Block Array [0...N]"]
    LBA -->|"Internal Mapping"| Controller["Disk Controller Hardware"]
    Controller -->|"Physical Controls"| Hardware["Platter / Track / Sector"]
```

*   **Advantage:** Simplifies OS drivers; internal drive electronics handle bad-block remapping and defect management transparently.
*   **Disadvantage:** Hides exact physical geometry, making it harder for the OS to optimize track placement.

---

## Disk Scheduling Policies

To minimize actuator arm movement, the OS orders pending disk I/O requests:

```mermaid
flowchart TD
    TITLE["Disk Scheduling Algorithms"]
    
    FIFO["<b>First-In, First-Out (FIFO)</b><br/>Processes requests strictly in arrival order.<br/>• <i>Pro:</i> Fair; zero starvation.<br/>• <i>Con:</i> High seek times; inefficient arm movement."]
    
    SSTF["<b>Shortest Seek Time First (SSTF)</b><br/>Serves the request closest to the current head position.<br/>• <i>Pro:</i> Reduces seek latency.<br/>• <i>Con:</i> Causes starvation for distant tracks."]
    
    SCAN["<b>Elevator (SCAN)</b><br/>Arm sweeps continuously back and forth across tracks.<br/>• <i>Pro:</i> Bounds maximum wait time; prevents starvation.<br/>• <i>Con:</i> Favors middle tracks over extreme edges."]
    
    TITLE --> FIFO
    TITLE --> SSTF
    TITLE --> SCAN
```

---

## Related Notes

- [[Solid State Drives & NAND Flash|Solid State Drives & NAND Flash]]
- [[RAID Architectures|RAID Architectures]]
- [[Virtual Memory & Address Translation Fundamentals|Virtual Memory & Address Translation Fundamentals]]