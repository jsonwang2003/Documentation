## Task
Implement the function `trim_end_spaces`
- Write a function that takes a `char*` string that may have **trailing spaces** and modifies the string *in-place* so that the first space in the list of trailing spaces is replaced with a null-terminator(`\0`)
- The function should not return anything; it should modify the input string directly
### Function Signature
```c
// Given a char* that may have trailing spaces, change the string to have a null terminator where the first space in the list of trailing spaces starts.
void trim_end_spaces(char* str);
```

---
## Example
```bash
$ ./trim_end_spaces
a b c    
a b c
hello world   
hello world
test
test
"   "
""
a    b    
a    b
```

---
## Code
```c
void trim_end_spaces(char* str) {
    int end = strlen(str) - 1;
    while(end >= 0 && str[end] == ' '){
	    end--;
    }
    str[end + 1] = 0;
}
```