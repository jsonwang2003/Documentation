Consider the following C code
```c
char str[] = "iCDuLNb";
char *p = str + 1;
printf("%s", p);
```
if the base address for `str` is `0x000081e0`, what will be printed when the `printf` statement is executed?
**Output**: `"CDuLNb"`

Additionally, what is the value of `p` in hexadecimal?
**p** = 0x000081e1

---
## Answer
When working with strings in C, remember that strings are null-terminated. This means that the end of a string is marked by the null character (`'\0'`), which tells functions like `printf` where to stop printing.  
  
In the given example, the pointer `p` is initialized as `str + 1`, meaning it points to the character at position `1` in the string `iCDuLNb`. The `printf("%s", p);` statement prints the substring starting from that character until the null character is encountered, giving us the Output seen above.  
  
Meanwhile, to find the address of `p`, we can calculate it as follows:  
1. The base address of `str` is `0x000081e0`.  
2. The offset to the character at index `1` is `1 * 1` (since each character is 1 byte).  
3. Therefore, the address of `p` is `0x000081e0 + 1`.