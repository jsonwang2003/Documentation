> [!ABSTRACT]
> 
> Managing strings in C often requires dynamic allocation because the length of input is rarely known at compile-time. By using a struct to wrap a heap-allocated char*, we can create persistent, flexible string objects.

### 1. Struct Alignment and Size
When defining a struct that contains a pointer, the compiler must ensure that members are correctly "aligned" in memory for efficient CPU access.

```c
typedef struct String {
    int len;         // 4 bytes
    char* contents;  // 8 bytes (on a 64-bit system)
} String;
```

- `sizeof(String)`: Even though $4 + 8 = 12$, `sizeof(String)` will often return **16**.
- **Alignment**: The `char* contents` member is 8 bytes long (since it is an address). C requires this member to start at an address that is a multiple of 8 (8-byte aligned). This results in 4 bytes of "padding" after the `int len`.

---
### 2. Implementing `readline`
In high-level languages like Python, reading input is handled automatically. In C, we use a temporary buffer on the **Stack** to capture input and then move it to a precisely sized block on the **Heap**.

```c
String readline(){
    char input[1000]; // Temporary stack buffer
    
    if (fgets(input, sizeof(input), stdin) == NULL) {
        return (String){0, NULL};
    }
    
    int length = strlen(input);
    
    // Allocate exactly enough space on the heap (+1 for null terminator)
    char* contents = malloc(length + 1); 
    strcpy(contents, input);
    
    String s = { length, contents };
    return s; // The struct is copied to the caller, but the 'contents' pointer still points to the heap.
}
```

- **Why Malloc?**: Using `malloc` ensures the string data persists after `readline` returns.
- **Copying Behavior**: When `s` is returned, the `len` and the `contents` **address** are copied to the caller's stack, but the actual text remains in the same spot on the heap.

---
### 3. The Warning: Memory Leaks
Because `malloc` does not automatically clean up, this implementation currently suffers from a **Memory Leak**.

```c
while(1){
    String s = readline(); // A new heap block is created every loop
    if(strstr(s.contents, to_find) != NULL){
        printf("%s\n", s.contents);
    }
    // WARNING: 's.contents' is never freed!
}
```

> [!WARNING]
> 
> Every call to readline allocates new memory. Once the loop repeats and the variable s is overwritten, the previous pointer is lost. Those heap blocks become unreachable, wasting memory until the program terminates.

---
### Key Takeaway
To fix this leak, the caller must manually call `free(s.contents)` once they are finished with the string. This ensures the heap space is reclaimed for future allocations.