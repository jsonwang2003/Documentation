> [!INFO]
> In C, strings are implemented as **arrays of `char` values**
> Not all character array is used as a **C string**, but every **C string** is a character array

For more information on [[Computer Science Introduction/Programming Languages/C/Syntax/Arrays|C Arrays]]

---
> [!RECALL]
> Arrays in C might be defined with a *larger size* than a program ultimately uses → we cannot assume that a string's length is equal to that of the array that stores it. 
> Therefore strings in C **must end with a special character, *null character* ('\0') to indicate the end of the string**

Strings that ends with a **null character** are said to be **null-terminated**
- Common source of errors
- To store the string "hi" → `char my_string[3] = {'h', 'i', '\0'};`

> [!IMPORTANT]
> Character arrays must be declared with enough capacity to store each character (including the null character)

---
## C String Libraries
- Contains functions for manipulating strings
- Needs to include the `string.h` header
- Library functions often locate **the end of the string by searching '\0'** or **add a '\0' character to the end of the string modified**
### Printing Strings
When using `printf` to print a string, use the `%s` placeholder in the format string → prints all the characters in the array argument until encounters the **null character**

### Example
```c
# include <stdio.h>
# include <string.h> // Include the C string library

int main(void){
	char str1[10];
	char str2[10];
	int len;
	
	str1[0] = 'h';
	str1[1] = 'i';
	str1[2] = '\0';
	
	len = strlen(str1);
	
	printf("%s %d\n", str1, len); // Prints hi 2
	
	strcpy(str2, "hello"); // Copy the string "hello" to str2
	len = strlen(str2);
	printf("%s has %d chars\n", str2, len); // prints: hello has 5 characters
}
```

- `strlen`: returns the number of characters in its string argument (excluding '\0')
- `strcpy`: copies **one character at a time** from a source string (second parameter) to a destination string (first parameter) until it reaches a **null character** in the source

> [!WARNING]
> Most C string library functions expect the call to **pass in a character array** that has **enough capacity for the function to perform its job**
> → Don't want to call `strcpy` with a destination string that **isn't large enough to contain the source** → leads to **undefined behavior in program**

> [!IMPORTANT]
> C string library functions **require string values passed to them are correctly formed with a terminating '\0' character**. Programmer's responsibility, not compiler's

> The `strcpy` call in the example is safe, but in general, `strcpy` poses a **security risk** because it assumes that its destination is **large enough to store the entire string** (may not always be the case)

#### String.h
- `int strlen(char* str)`: return the length of the string
- `void strcpy(char* dest, char* src)`: copies characters from `src` to `dest`
- `void strcat(char* dest, char* src)`: append `src` to the **end** of `dest`

> `char[] str` vs `char* str`
> **In function arguments these do the same thing**
### C's Support for Statically Allocated Strings



