---
title: File Systems
description: A directory covering file system architectures, disk layouts, buffer caches, reliability, and fundamental I/O operations.
aliases:
  - File Systems Directory
  - File Systems Hub
  - File System Index
tags:
  - index
  - operating-systems
  - file-system
  - storage
---

> [!abstract] Overview
> A File System is the operating system abstraction that converts raw, unformatted physical block storage (HDDs, SSDs, NVMe drives) into a structured, persistent, and human-readable hierarchy of files and directories. It handles logical-to-physical block mapping, metadata tracking, access control, and crash recovery.

---
# Core Modules

| Note Link | Description | Key Concepts |
| :--- | :--- | :--- |
| [[File Systems & Storage Technologies]] | Overview of persistent storage hardware characteristics and block-device requirements. | Sector/Block Addressing, HDDs, SSDs, Flash Memory |
| [[File System Layout]] | Structural organization of disk space into operational regions and superblock headers. | Boot Block, Superblock, Inode Tables, Data Blocks, Bitmaps |
| [[Common File Operations]] | VFS and system call interfaces handling kernel state management and process descriptors. | FDs, System-Wide Open File Table, Inode Caching, VFS |
| [[Multi-Level Indexed Layout]] | Unix-style inode indexing models for direct and indirect block allocation. | Direct Pointers, Single/Double/Triple Indirect Pointers, Inodes |
| [[Optimized Disk Layout]] | Strategies for maximizing sequential I/O throughput and reducing seek latency. | Cylinder Groups, FFS, Extents, Spatial Locality |
| [[File Buffer Cache]] | In-memory caching structures bridging RAM and persistent storage latency gaps. | Page Cache, Write-Back, Write-Through, LRU, Read-Ahead |
| [[File System Reliability]] | Crash consistency, transaction logging, and fault-tolerance mechanisms. | Journaling, Write-Ahead Logging, Copy-on-Write, fsck, RAID |


---
# Related Modules

- **[[Computer Systems/System Programming/Memory Management/index|Memory Management]]**: Integration of the page cache with virtual memory via unified buffer caches, page-fault handling for file-backed pages, and zero-copy streaming (`mmap`, `sendfile`).
- **[[Computer Systems/Operating Systems/Storage & IO Systems/index|Storage & IO Systems]]**: Lower-level block I/O layer, Direct Memory Access (DMA) engine execution, interrupt handling, and disk request scheduling (SSTF, SCAN, C-SCAN).
- **[[Computer Systems/Operating Systems/Kernel & Architecture/index|Kernel Architecture]]**: Process state transitions during blocking I/O, file descriptor table inheritance across `fork()`, and reader-writer concurrency control for directory caches.