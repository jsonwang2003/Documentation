> [!ABSTRACT]
> 
> In C, strings are null-terminated arrays of characters. Because arrays cannot be easily resized or returned from functions, we must carefully manage "Out Parameters" and understand the relationship between pointers and array memory.

### 1. The Out Parameter Pattern
In your `concat` implementation, `result` is an **Out Parameter**.
- Instead of the function creating and returning a string, the **caller** provides a pre-allocated space.
- The function then "fills in" that space.

**Memory Allocation Calculation:**
To safely concatenate strings $a$ and $b$, the result buffer must be:

```c
strlen(a) + strlen(b) + 1
```

The `+1` is mandatory to store the Null Terminator (`\0`), which marks the end of the string. Without it, `printf` would keep reading into adjacent memory, causing a "buffer over-read."

---
### 2. Why "Return `char[]`" Fails
You observed that returning a local array from a function causes a compilation error (or a crash).

```c
char[] concat(char a[], char b[]) {
    char result[alen + blen + 1]; // Allocated on the STACK
    // ... logic ...
    return result; // Error: result is destroyed when function returns!
}
```

> [!DANGER]
> 
> Dangling Pointers: Local variables (including arrays) live on the Stack. When a function returns, its stack frame is reclaimed. Any pointer to result now points to "garbage" memory that could be overwritten at any second. This is why we must use the Out Parameter pattern or the Heap (via malloc).

---
### 3. Command Line Arguments: `char** argv`
The entry point of a C program, `main`, receives arguments from the Operating System:
- **`argc`**: The number of arguments.
- **`argv`**: An array of pointers (a pointer to a pointer). Each element `argv[i]` is a `char*` pointing to a string on the stack.

**Memory Trace from example:**

```c
void concat(char* a, char* b, char* result){
	printf("a: %p, b: %p\nresult: %p\n", a, b, result);
	int alen = strlen(a), blen = strlen(b);
	for(int i = 0; i < alen; i += 1){
		result[i] = a[i];
	}
	for(int i = 0; i < blen; i += 1){
		result[alen + i] = b[i];
	}
	result[alen + blen] = 0;
}

int main(int argc, char** argv){
	printf("Argv: %p\n" argv);
	printf("Arg index 0: %p: %s\n", argv[0], argv[0]);
	printf("Arg index 1: %p: %s\n", argv[1], argv[1]);
	printf("Arg index 2: %p: %s\n", argv[2], argv[2]);
	
	char result[strlen(argv[1]) + strlen(argv[2]) + 1];
	concat(argv[1], argv[2], result);
	
	printf("%s\n", result);
}
```

When running `./args "Hello " "CSE29"`:
- `argv[0]`: Points to the program name (`./args`).
- `argv[1]`: Points to `"Hello "`.
- `argv[2]`: Points to `"CSE29"`.

```bash
$ gcc args.c -o args
$ ./args "Hello " "CSE29"
Argv: 0x7ffcda761c38
Arg index 0: 0x7ffcda7626e8: ./args
Arg index 1: 0x7ffcda7626ef: Hello
Arg index 2: 0x7ffcda7626f6: CSE29
a: 0x7ffe600146ef, b: 0x7ffe600146f6
result: 0x7ffe600130d0
Hello CSE29
```

Note that the addresses for `argv` elements (e.g., `...6e8`, `...6ef`) are very close to each other, as the OS packs these strings together in a special segment of the stack.

---
### 4. Pointer vs. Array Arguments
In function signatures, `char a[]` and `char* a` are identical to the compiler. Both represent the **address of the first character**.

```c
void concat(char* a, char* b, char* result)
```

Inside this function, `result[i]` is just shorthand for `*(result + i)`. Whether you passed a fixed array or a pointer to the heap, the logic remains the same: you are writing bytes to a specific memory location.

--- 
### Summary Table: String Sizes

|**String**|**strlen()**|**sizeof()**|**Buffer Needed**|
|---|---|---|---|
|`"Hello "`|6|7|7 bytes|
|`"CSE29"`|5|6|6 bytes|
|**Combined**|11|N/A|**12 bytes**|
