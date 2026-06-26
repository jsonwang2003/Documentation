> [!ABSTRACT]
> 
> Every variable in a C program is stored at a specific numerical address in the process's Address Space. We use the "address-of" operator (&) to find these locations and sizeof to measure the footprint they occupy on the Stack.

```c
void wheresmystuff(char* s){
	int x = 12;
	uint8_t y = 53;
	char z = 9;
	double ns[] = {4.0, 3.0, 9.0};
	
	printf("x=%d, %lu bytes, starts as: %p\n", x, sizeof(x), &x);
	printf("y=%hhu, %lu bytes, starts at: %p\n", y, sizeof(y), &y);
	printf("z=%hhu, %lu bytes, starts at: %p\n", z, sizeof(z), &z);
	printf("ns=[%f,%f,%f], %lu bytes, starts at: %p\n", ns[0], ns[1], ns[2], sizeof(ns), &ns);
	
	printf("s=\"%s\"@%p, %lu bytes, starts at: %p\n", s, s, sizeof(s), &s);
}

int main(){
	char str[] = "14 char string";
	wheresmystuff(str);
	printf("\nstr takes up %lu bytes starting at: %p\n", sizeof(str), &str);
}
```
### 1. Pointers and Addresses
A **pointer** is simply a variable that holds a memory address.
- **`&v`**: The "address-of" operator. It tells the OS: "Give me the memory location where `v` is stored."
- **`*`**: When used in a type (e.g., `int*`), it denotes a pointer. When used on a variable (e.g., `*ptr`), it **dereferences** the pointer to access the value at that address.

---
### 2. Measuring Memory with `sizeof`
The `sizeof(v)` operator returns the total number of bytes allocated for variable `v`. As seen in the `wheresmystuff` execution, these sizes are strictly defined by the data type:

|**Variable Type**|**Bytes**|**Explanation**|
|---|---|---|
|`int x`|**4**|Standard 32-bit integer.|
|`uint8_t y`|**1**|Unsigned 8-bit integer (exactly one byte).|
|`double ns[3]`|**24**|3 doubles $\times$ 8 bytes each = 24 bytes.|
|`char* s`|**8**|On a 64-bit system, all addresses are 8 bytes long.|

---
### 3. The Array vs. Pointer Distinction
The most critical observation from your code is the difference between `str` (the array) and `s` (the pointer argument).
#### The Array (`str`)
- Declared as: `char str[] = "14 char string";`
- **Size**: 15 bytes (14 characters + 1 for the null terminator `\0`).
- **Address**: `0x16b13aee8`.
- When you call `sizeof(str)`, you get the size of the **entire string content**.

#### The Pointer Argument (`s`)
- Declared as: `wheresmystuff(char* s)`
- **Size**: 8 bytes.
- **Address of pointer**: `0x16b13ae98`.
- **Value of pointer**: `0x16b13aee8`.
- When `str` is passed into the function, it "decays" into a pointer. Inside the function, `s` is just a local variable on the stack that stores the **address** of the original string. Calling `sizeof(s)` only tells you the size of the address itself (8 bytes), not the string it points to.

---
### 4. Stack Layout Observations

```bash
$ gcc -Wall wheresmystuff.c -o wheresmystuff
$ ./wheresmystuff
x=12 takes up 4 bytes starting at: 0x16b13ae94
y=53 takes up 1 bytes starting at: 0x16b13ae93
z=9 takes up 1 bytes starting at: 0x16b13ae92
ns=[4.000000,3.000000,9.000000] takes up 24 bytes starting at: 0x16b13aea0

s="14 char string"@0x16b13aee8, 8 bytes, starts at: 0x16b13ae98

str takes up 15 bytes starting at: 0x16b13aee8
```

If you look closely at the addresses from your output:
- `x` is at `...e94`
- `y` is at `...e93`
- `z` is at `...e92`

The addresses are decreasing. This confirms that on most modern systems, the **Stack grows downwards** toward lower memory addresses as new local variables are allocated.

---
### Module Navigation
- **[[Pointer Operations]]**: How to do math with these hex addresses.
- **[[Size of Struct]]**: How `sizeof` works when we group different types together.
- **[[Implementation of Malloc]]**: How we get addresses on the **Heap** instead of the **Stack**.