---
title: Debuggers
---
## Overview

Debuggers and profiling tools are essential for identifying logical errors, memory leaks, and performance bottlenecks within a program. This collection covers tools used for low-level inspection and runtime analysis.

---
## 1. Core Debugging
Tools used to stop execution, inspect state, and trace code execution line-by-line.
- **[[GNU Debugger (GDB)]]**: The standard portable debugger for Unix-like systems. It allows you to see what is going on inside a program while it executes or what it was doing at the moment it crashed.
- _Key uses_: 
	- Breakpoints
	- backtracing
	- variable inspection.

---
## 2. Memory Analysis
Tools focused on the management and safety of system memory.
- **[[Valgrind]]**: An instrumentation framework for building dynamic analysis tools. It is primarily used for detecting memory-related bugs.
- _Key uses_: 
	- Detecting memory leaks (`memcheck`)
	- use-after-free errors
	- uninitialized memory access
---
## 3. Performance Profiling
Tools used to analyze program execution time and identify "hotspots" where the code is slow.
- **[[GPROF]]**: A hybrid instrumentation/sampling profiler for Unix programs.
- _Key uses_: 
	- Analyzing call graphs 
	- Determining which functions are consuming the most CPU time to guide optimization.
