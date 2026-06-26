---
title: Program Assessments
---
> [!ABSTRACT]
> 
> The Program Assessments (PAs) are the capstone projects of the course. Unlike the smaller problem sets, these require integrating multiple concepts to build cohesive, industry-standard system tools and applications.

---
## The Projects

### [[Practice Examples/Program Assessments/UTF-8|1. UTF-8 Analyzer]]
- **Focus**: Multi-byte Data Representation and Character Validation.
- **Goal**: Develop a comprehensive analyzer that processes UTF-8 strings to extract metadata, transform content, and identify specific Unicode ranges.
- **Key Challenges**:
    - Implementing **variable-width parsing** to calculate logical string length and byte-per-codepoint distribution.
    - Performing **bit-level encoding** to convert multi-byte sequences into decimal codepoint values.
    - Applying **range-based filtering** to detect specific symbols, such as identifying animal emojis ($U+1F400$ to $U+1F43F$).
    - Managing **memory-safe substrings** and ASCII-only transformations (capitalization) within a mixed-encoding buffer.
#### Design Questions
- **UTF-8 vs. UTF-32**: A trade-off between **Memory Efficiency** (UTF-8's variable width saves space for ASCII) and **Computational Simplicity** (UTF-32's fixed 4-byte width allows $O(1)$ random access).
- **The "Self-Synchronizing" Property**: The leading `10` bits in continuation bytes are not wasteful; they make UTF-8 **self-synchronizing**. This allows a program to determine if it is at the start of a character in $O(1)$ time without scanning from the beginning of the string, and prevents data corruption from misaligning a single byte.

---
### [[Practice Examples/Program Assessments/Hashing and Passwords|2. Password Cracker (SHA256)]]
- **Focus**: Cryptographic Hashing and Brute-Force Heuristics.
- **Goal**: Build a utility that identifies a plaintext password from a provided SHA256 hex hash by processing a stream of potential guesses from `stdin`.
- **Key Challenges**:
    - Integrating the **OpenSSL `lcrypto` library** to perform deterministic SHA256 hashing.
    - Converting raw **32-byte binary hashes** into **64-character hexadecimal strings** for comparison.
    - Implementing **single-mutation variants**: For every guess, the program automatically generates and tests every version where a single ASCII character’s case is flipped.
    - Managing **persistent stream processing** to efficiently handle large dictionary files without excessive memory overhead.
#### Design Questions
- **Cracking Techniques**:
	- **Rainbow Tables**: Using precomputed tables of (Password, Hash) pairs to bypass the need for computation at runtime (trading storage for speed).
	- **Spidering**: Generating a targeted dictionary based on specific personal data or keywords related to the target user.
- **The Bottleneck**: Because the memory footprint of a cracker is tiny (storing only a few strings), it is **CPU-bound** (computationally limited). The sheer number of possible combinations means the speed of the hash function and the processor is the primary constraint.

---
### [[Practice Examples/Program Assessments/The Pioneer Shell|3. The Pioneer Shell (pish)]]

- **Focus**: Process Control, System Calls, and Shell Architecture.
- **Goal**: Build a simplified Unix-like shell capable of operating in both **Interactive Mode** (user input) and **Batch Mode** (script files).
- **Key Challenges**:
    - **Process Lifecycle**: Utilizing `fork()` to spawn child processes, `execvp()` to replace the process image with external programs, and `waitpid()` to synchronize execution.
    - **Command Parsing**: Implementing a robust tokenizer using `strtok` that handles arbitrary whitespace and tabs to build an `argc`/`argv` structure.
    - **Built-in Commands**: Manually managing the process environment with `chdir()` for the `cd` command and implementing controlled termination for `exit`.
    - **Persistent History**: Managing file I/O with `fopen`, `fprintf`, and `fgets` to store and retrieve command history from a hidden file (`~/.pish_history`).
    - **Error Diagnostics**: Integrating standard error reporting using `perror()` to debug failed system calls like file access or execution errors.
#### Code Mechanics
1. **Read**: The shell captures a line of input.
2. **Parse**: The input is broken into an argument vector.
3. **Evaluate**:
    - If the command is **Built-in** (`cd`, `exit`, `history`), the shell calls a function within its own code to modify its state (like changing its own current directory).
    - If the command is an **External Program** (`ls`, `gcc`), the shell must `fork` a new copy of itself. The "child" copy calls `execvp` to become the new program, while the "parent" waits for it to finish.

---
### [[Practice Examples/Program Assessments/Malloc|4. Custom Heap Allocator (vmalloc)]]
- **Focus**: Dynamic Memory Management and Alignment.
- **Goal**: Implement a robust memory allocation library (`vmalloc` and `vmfree`) that manages a simulated heap using custom headers, footers, and metadata encoding.
- **Key Challenges**:
    - **Best-Fit Policy**: Traversing the implicit free list to find the smallest available block that satisfies a request, minimizing internal fragmentation.
    - **Splitting & Coalescing**: Dynamically dividing large free blocks during allocation and merging adjacent free blocks during deallocation to optimize space.
    - **16-Byte Alignment**: Ensuring that every returned pointer is a multiple of 16, requiring careful padding and rounding of block sizes.
    - **Bitwise Metadata**: Using the least significant bits of the `size_status` field to track block states (Busy vs. Free) and the state of the preceding block (Prev-Busy).
    - **Pointer Arithmetic**: Safely navigating the heap by casting between `struct block_header *`, `uint64_t *`, and `char *` to perform byte-accurate jumps.
#### Implementation Mechanics Summary
Utilizes several advanced C techniques to maintain the heap:
1. **Implicit Free List**: The heap is essentially a giant array of blocks where each header tells you how many bytes to "jump" to find the next header.
2. **The Header/Footer Sandwich**:
    - **Headers** allow forward traversal.
    - **Footers** (only in free blocks) allow backward traversal, which is essential for coalescing with the _previous_ block in $O(1)$ time.
3. **Boundary Tagging**: By using the LSB of the size field, you store state information without consuming extra memory, a technique used in production-grade allocators like `dlmalloc`.

---
### [[Practice Examples/Program Assessments/Web Server|5. Persistent Chat Server (HTTP)]]
- **Focus**: Network Programming, HTTP Protocol, and Data Persistence.
- **Goal**: Develop a multi-user chat server that processes HTTP GET requests to post messages, list conversation history, and add reactions.
- **Key Challenges**:
    - **Socket Communication**: Utilizing `send()` and `recv()` to handle live TCP/IP connections and managing the server lifecycle.
    - **Protocol Parsing**: Extracting meaningful data from raw HTTP request strings, including URL-decoding (e.g., `%20` to spaces) and query parameter extraction (`user=`, `message=`, `id=`).
    - **Dynamic State Management**: Using an array of structs to store chats and nested reaction arrays, ensuring data persists across multiple client connections.
    - **Formatted Output**: Generating standardized, human-readable plain text responses that include formatted timestamps and aligned chat/reaction lines.
    - **Error Handling**: Implementing appropriate HTTP status codes (400, 404, 500) to respond to invalid IDs, missing parameters, or length violations.
#### Implementation Mechanics Summary
1. **Request Routing**: The server acts as a "dispatcher." It reads the first line of the HTTP request to decide whether to trigger `handle_post`, `handle_reaction`, or `respond_with_chats`.
2. **String Sanitization**: Since URLs transmit data in a specific format, the `url_decode` function is critical to ensure that characters like `+` or `%21` are converted back to their original text before being stored.
3. **Timestamping**: By using `<time.h>` and `strftime`, the server marks every message with a permanent "wall clock" time, which is essential for the chronological rendering of the chat history.

---
## Integration Map

| **Project**    | **Primary Skill**  | **Secondary Skill** |
| -------------- | ------------------ | ------------------- |
| **UTF-8**      | Bitwise Operations | Character Encoding  |
| **Passwords**  | Hashing            | Data Integrity      |
| **Shell**      | Process Control    | String Tokenization |
| **Malloc**     | Pointer Arithmetic | Heap Architecture   |
| **Web Server** | Socket Programming | HTTP Protocol       |

---
## Related Toolkits
- **[[Syntax/index|C Syntax]]**: The mechanical foundation for all implementations, especially complex data structures and pointers.
- **[[Practice Examples/Lecture Examples/index|Lecture Examples]]**: Provides the high-level architecture designs (like the Heap Structure or the Client-Server model) used in these assessments.
- **[[Discrete Structures/Discrete Algorithms/index|Algorithm Analysis]]**: Critical for ensuring that your `malloc` search or your shell's parsing logic remains efficient under load.