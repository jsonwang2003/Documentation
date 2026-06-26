---
title: Problem Sets 3
---
> [!ABSTRACT]
> 
> This problem set focuses on the interaction between a user-space program and the Operating System kernel. You will practice the "Fork-Exec-Wait" pattern, manage file I/O, and implement the string parsing logic required to build a functional command-line interface.

## 1. Process Management & System Calls
The core mechanics of how a shell launches and manages programs.
- **[[Fork and Wait]]**: Mastering the creation of child processes and parent synchronization.
- **[[Running ls Command]]**: Using the `exec` family to replace a process image with an external binary.
- **[[Run CD]]**: Implementing "Built-in" commands that modify the shell's own state (like the current directory).
- **[[Run Exit]]**: Handling clean termination of a session.
- **[[Test Perror]]**: Using `perror` to translate system error codes into human-readable strings.

---
## 2. Command Line Parsing
Turning raw user input into actionable data for system calls.
- **[[Tokenize Args]]** & **[[Parse Args]]**: Converting a single command string into an array of pointers (`char** argv`).
- **[[Parse Flags]]**: Identifying and separating command options (e.g., `-l`, `-a`).
- **[[Split Space]]**: General utility for breaking strings based on whitespace.
- **[[Print First Word Until Exit]]**: Basic loop structure for a Read-Eval-Print Loop (REPL).

---
## 3. File I/O and String Utilities
Interacting with the file system and cleaning up data for display.
- **[[Find In File]]**: Searching for patterns within a file stream.
- **[[Write With Line Numbers]]**: Formatting output while writing to a file.
- **[[Get Line]]**: Safely reading a line of arbitrary length from a stream.
- **[[Read History]]** & **[[Write History]]**: Persistent storage for shell command history.
- **[[Trim Trailing Spaces from a String]]**: Sanitizing input before processing.
- **[[Count Occurrences of a Character in a String]]**: Utility for analyzing command syntax or delimiter frequency.

---
## 4. Advanced Formatting
- **[[Print Parts Tabs]]**: Managing tab-separated values.
- **[[Print Space-Separated Parts of a String]]**: Breaking down complex output into readable sections.
- **[[Extract Second]]**: Targeted extraction of specific arguments or tokens.

---
### Implementation Mapping: The Shell Architecture

| **Shell Task**       | **Relevant Problem Set File**                          |
| -------------------- | ------------------------------------------------------ |
| **Read Input**       | `Get Line`, `Trim Trailing Spaces...`                  |
| **Tokenize**         | `Tokenize Args`, `Split Space`, `Count Occurrences...` |
| **Identify Command** | `Parse Flags`, `Run CD` (Built-ins)                    |
| **Execute**          | `Fork and Wait`, `Running ls Command`                  |
| **Error Handle**     | `Test Perror`                                          |