---
description: "Performance, latency, bandwidth, capacity, and cost trade-offs across DRAM, Non-Volatile Memory (NVM), Solid State Disks (SSDs), and Hard Disk Drives (HDDs)."
aliases:
  - Memory Hierarchy Spectrum
  - Storage Spectrum
  - NVM vs SSD vs HDD
  - Memory Technologies
tags:
  - operating-systems
  - storage
  - memory-hierarchy
  - hardware
---
> [!abstract] Abstract
> Modern computing architectures employ a multi-tiered **Memory and Storage Hierarchy** to balance execution speed, persistent capacity, and financial cost. While volatile **DRAM** provides sub-microsecond access times for active execution, non-volatile media (**NVM**, **SSDs**, and **HDDs**) retain data across system power cycles at varying latency and cost profiles.
> 
> - **Category:** System Memory & Storage Hardware
> - **Key Trade-off:** Access Latency vs. Cost per Byte
> - **Volatility Boundary:** DRAM (Volatile) vs. NVM / SSD / HDD (Non-Volatile Persistence)

---

## The Storage & Memory Hierarchy Spectrum

Hardware storage technologies range from ultra-low-latency volatile CPU registers down to high-capacity, low-cost mechanical disks:

![[Pasted image 20260731133617.png]]

---

## Technical Parameter Comparison

| Technology Tier | Volatility | Access Latency | Throughput Bandwidth | Typical Capacity | Relative Cost |
|---|---|---|---|---|---|
| **DRAM** | Volatile | $50 \text{ to } 100 \text{ ns}$ | $50 \text{ to } 100 \text{ GB/s}$ | Tens of GB / module | Highest ($\$\$\$\$\$\$$) |
| **Non-Volatile Memory (NVM)** | Non-Volatile | $\sim 300 \text{ ns}$ | A few $\text{GB/s}$ | Tens to hundreds of GB / module | High ($\$\$\$\$$) |
| **Solid State Disks (SSD)** | Non-Volatile | $30 \text{ to } 100 \ \mu\text{s}$ | A few $\text{GB/s}$ | $64 \text{ GB to } 4 \text{ TB}$ | Moderate ($\$\$\$$) |
| **Hard Disk Drives (HDD)** | Non-Volatile | $5 \text{ to } 10 \text{ ms}$ | $100 \text{ to } 150 \text{ MB/s}$ | $1 \text{ to } 8 \text{ TB}$ | Lowest ($\$\$$) |

---

## System Integration & Design Implications

1.  **DRAM vs. Secondary Storage:** Because CPU execution speeds require sub-microsecond memory responses, operating systems treat DRAM as temporary scratchpad memory, utilizing virtual memory mechanisms like [[Demand Paging & Page Faults|Demand Paging]] to swap idle frames out to SSD or HDD backing stores.
2.  **Emerging NVM Architecture:** Persistent Non-Volatile Memory bridges the gap between DRAM and flash SSDs by offering near-DRAM read/write speeds while maintaining persistent data across system reboot cycles without block-erase restrictions.

---

## Related Notes

- [[Hard Disk Drive Mechanics & Scheduling|Hard Disk Drive Mechanics & Scheduling]]
- [[Solid State Drives & NAND Flash|Solid State Drives & NAND Flash]]
- [[Virtual Memory & Address Translation Fundamentals|Virtual Memory & Address Translation Fundamentals]]