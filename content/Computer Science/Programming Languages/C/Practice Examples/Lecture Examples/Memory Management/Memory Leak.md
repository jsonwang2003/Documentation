> [!ABSTRACT]
> 
> A Memory Leak occurs when a program allocates memory on the heap but fails to release it. Over time, this "leaked" memory accumulates, potentially causing the program to crash or slow down the entire system.

### 1. The Role of `free()`
The `free(void* ptr)` function tells the Operating System that the memory at that address is no longer needed.
- **Mechanism**: It marks the memory block as "Available" so that subsequent calls to `malloc` can reuse that same physical space.
- **Address Reuse**: In your code, after adding `free(s.contents)`, you will notice the printed addresses stay the same. This is because the allocator is immediately reclaiming the space from the previous loop iteration.

---
### 2. Implementation: Solving the `readline` Leak
In the original loop, every line of text stayed on the heap forever. By adding a single line, we ensure the program's memory footprint remains constant regardless of how many lines are read.

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define READLINE_LENGTH 4096

typedef struct String{
	int len;
	char* content;
} String;

String readline(){
	char input[READLINE_LENGTH];
	char* result = fgets(input, READLINE_LENGTH, stdin);
	if(result == NULL){
		String s = {-1, NULL};
		return s;
	}
	int length = strlen(input);
	char* contents = malloc(length + 1);
	strcpy(contents, input);
	String s = {length, contents};
	return s;
}

int main(int argc, char** argv){
	char* to_find = argv[1];
	while(1){
		String s = readline();
		if(s.contents == NULL) {break;}
		printf("%p (%d)\n", s.contents, s.len);
		if(strstr(s.contents, to_find) != NULL){
			printf("%s\n", s.contents);
		}
		
		// added
		free(s.contents);
	}
}
```

---
### 3. The Dangers of `free()`: Undefined Behavior
Managing memory manually is notoriously difficult. If you violate the rules of `free()`, your program enters a state of **Undefined Behavior (UB)**, where the OS may crash the program (Segmentation Fault) or silent data corruption may occur.

#### The Four "Golden Rules"
1. **Origin**: The pointer must have been returned by `malloc`, `calloc`, or `realloc`. You cannot `free` a stack address.
2. **Non-NULL**: While `free(NULL)` is technically safe (it does nothing), freeing an uninitialized pointer is fatal.
3. **Double Free**: You must not call `free()` on the same address twice. This corrupts the allocator's internal metadata.
4. **Use After Free**: Once you `free(ptr)`, you must never access that memory again. The data there is now "garbage" or belongs to another part of the program.

> [!QUOTE]
> 
> "It is inhumanly possible to use free() correctly in large programs."
> 
> This difficulty is why modern languages (Java, Python, Go) use Garbage Collection, and why C programmers use tools like Valgrind to track leaks.

---
### 4. Memory Management Summary

|**Action**|**Result if Forgotten**|**Result if Done Wrong**|
|---|---|---|
|**`malloc`**|No memory to use|Possible crash (check for NULL)|
|**`free`**|**Memory Leak**|**Undefined Behavior** / Crash|

---
### Module Navigation
- **[[Coalescing Heap]]**: How the allocator actually merges these freed blocks.
- **[[Virtual Memory]]**: How the OS maps these heap addresses to physical RAM.
- **[[Dynamic Struct]]**: Reviewing the structure that holds these heap pointers.