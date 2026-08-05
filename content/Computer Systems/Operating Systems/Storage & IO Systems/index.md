---
title: "Storage & IO Systems"
description: "A directory covering persistent physical storage devices, mechanical HDDs, NAND flash SSDs, disk scheduling, RAID architectures, file system abstractions, permissions, and memory technology trade-offs."
aliases:
  - Storage Index
  - Storage & IO Hub
  - Storage Systems
tags:
  - index
  - operating-systems
  - storage
  - io-systems
---
> [!abstract] Overview
> **Storage & IO Systems** manage non-volatile persistent media, bridging the speed gap between high-speed CPU registers/RAM and slow secondary storage. The operating system hides physical hardware complexities (such as spinning magnetic platters, mechanical seeks, and NAND flash erase blocks) behind standardized abstractions like the **Logical Block Interface** and byte-oriented **File Systems**.

---

## Module Notes

- [[File Systems & Storage Technologies|File Systems & Storage Technologies]]
- [[Hard Disk Drive Mechanics & Scheduling|Hard Disk Drive Mechanics & Scheduling]]
- [[Solid State Drives & NAND Flash|Solid State Drives & NAND Flash]]
- [[RAID Architectures|RAID Architectures]]

---

## Physical Storage & Abstraction Hierarchy

```mermaid
graph TD
    Root["Storage Media Architectures"]

    subgraph Mechanical["Mechanical Storage"]
        HDD["<b>Hard Disk Drives (HDD)</b><br/>Magnetic platters, rotating spindles, and mechanical arms.<br/>• Bounded by seek time and rotational latency."]
    end

    subgraph Flash["Semiconductor Flash"]
        SSD["<b>Solid State Drives (SSD)</b><br/>Non-volatile NAND flash chips managed by an onboard FTL.<br/>• Fast random access; asymmetric block-erase constraints."]
    end

    subgraph Redundant["Multi-Drive Arrays"]
        RAID["<b>RAID Configurations</b><br/>Combines multiple physical disks into a unified logical drive.<br/>• Striping (Speed), Mirroring (Reliability), Parity (Fault Recovery)."]
    end

    subgraph Abstraction["Logical File Systems & Hierarchy"]
        FS["<b>File System Abstractions & Memory Spectrum</b><br/>VFS/IFS layer, byte streams, access patterns, protection bits, and hardware latency comparison."]
    end

    Root --> Mechanical
    Root --> Flash
    Root --> Redundant
    Root --> Abstraction
```

---

## Related Modules

- [[Computer Systems/Operating Systems/Memory Management/index|Memory Management Subsystem]]
- [[Interrupts and Exceptions|Interrupts and Exceptions]]
- [[Computer Systems/Operating Systems/index|Operating Systems Main Directory]]