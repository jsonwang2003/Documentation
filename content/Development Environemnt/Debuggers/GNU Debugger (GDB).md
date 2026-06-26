> [!INFO] Source Attribution
> These notes are synthesized from the **[GNU Debugger (GDB) Tutorial](https://www.tutorialspoint.com/gnu_debugger/index.htm)** hosted by Tutorialspoint.
## What is GNU Debugger (GDB)?

GDB is a portable debugger that runs on many Unix-like systems. It allows you to see what is going on "inside" another program while it executes—or what that program was doing at the moment it crashed.
### Key Capabilities

- **Crash Analysis:** Determine exactly which statement or expression caused a "core dump" (crash).
- **Variable Inspection:** Examine and modify values of variables or results of expressions at any point during execution.
- **Execution Control:** Run programs up to a specific point, or step through them line by line to observe state changes.
- **Call Stack Analysis:** Trace the history of function calls (backtrace) to see how the program reached its current state.

> [!CAUTION] Limitations
> 
> - **Compilation:** GDB cannot debug programs that fail to compile.
>     
> - **Memory Leaks:** While GDB can help find bugs that _lead_ to leaks, it is not a dedicated leak detection tool (use **Valgrind** for this).
>     

---
## Installation & Prerequisites

### 1. Prerequisites
- **Compiler:** An ANSI-compliant C compiler (like `gcc`).
- **Space:** ~115 MB to build from source; ~20 MB for a standard installation.
### 2. Installation Command (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install libc6-dbg gdb valgrind
```

### 3. Verify Installation

```Bash
gdb --version
# or
gdb -help
```
---
## The Debugging Symbol Table

To debug a program effectively, GDB needs a **Symbol Table**. This maps the machine code in the binary back to your source code (variable names, function names, and line numbers).
### Compiling for Debugging
Standard compilation (`gcc hello.c -o hello`) strips away this information to save space. To include the symbol table, you **must** use the `-g` flag:

```Bash
gcc -g hello.cc -o hello
```

- **Note:** Debug builds are larger and slower because they contain this extra metadata.
- **Versioning:** If you change your code, you must re-compile to generate a new, matching symbol table.

---
## Core GDB Command Reference

### Starting and Stopping

|**Command**|**Shorthand**|**Description**|
|---|---|---|
|`gdb a.out`|-|Start GDB with the specified executable.|
|`run`|`r`|Start the program (optionally add arguments: `r arg1`).|
|`quit`|`q`|Exit GDB.|
|`help`|`h`|List all help topics or a specific command (`h break`).|
### Execution Control (Stepping)

|**Command**|**Shorthand**|**Description**|
|---|---|---|
|`list`|`l`|Show 10 lines of source code.|
|`next`|`n`|Run the next line; **skips over** function calls.|
|`step`|`s`|Run the next line; **steps into** function calls.|
|`finish`|`f`|Finish the current function and return to the caller.|
|`continue`|`c`|Resume execution until the next breakpoint.|
### Breakpoints & Watchpoints
Breakpoints stop execution at a location. Watchpoints stop execution when a **data condition** changes.
- `break 45`: Stop at line 45.
- `break my_func`: Stop at the entry of `my_func`.
- `watch x == 3`: Stop whenever the condition `x == 3` becomes true or false.
- `info break`: List all current breakpoints.
- `delete N`: Delete breakpoint number `N`.

### Inspecting & Modifying Variables
- `print var`: Display the current value of `var`.
- `display var`: Automatically print `var` every time the program stops.
- `set var = 10`: Change the value of a variable during runtime to test different scenarios.
- `call func()`: Execute a function manually from the prompt.

---
## Navigating the Call Stack

The stack shows the path of function calls that led to the current point. A **Frame** represents one specific function call on that stack.
- `bt` (Backtrace): Shows the entire call stack.
- `up` / `down`: Move focus between the calling and called functions to inspect local variables in different scopes.


![Image of a function call stack with frames](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTJdwA15v2pyYoEF09JP2K9UaR2QOfsXLPWy8ocrqZYUakE46vvIkQO3E3RkfDlbEo_hh7vlED-Jh4wgtQq_VWZ9vAg7k6eB_a2sd8dY9lIfVmPPKE)

### Example Backtrace Output:

```txt
#0  c_func() at source.c:10
#1  0x08048456 in b_func() at source.c:20
#2  0x080484af in a_func() at source.c:30
#3  0x080483fe in main() at source.c:40
```
---
## Signal Handling
Signals (like `SIGSEGV` for segmentation faults or `SIGINT` for interrupts) can be managed within GDB.
- `handle [signal] [action]`: Tell GDB how to react (e.g., `handle SIGUSR1 nostop` to prevent GDB from pausing when that signal is received).

---
## GDB Cheat Sheet
![[GDB Cheat Sheet.pdf]]


