## Task
Implement the function `count_occurrences`
- Takes a `char* str` and a `char to_count`
- Return the number of times `to_count` appears in `str`
- Should return an `int` with the count
### Function Signature
```c
// Returns the number of times the to_count character appears in str
int count_occurrences(const char* str, char to_count);
```

---
## Example
```bash
$ ./count_occurrences
hello l
2
banana a
3
test t
2
test x
0
"   "  
3
"" a
0
a a
1
a b
0
```

---
## Code
```c
int count_occurrences(const char* str, char to_count) {
    int count = 0;
    for(int i = 0; str[i] != 0; i++){
        count += (str[i] == to_count) ? 1 : 0;
    }
    return count;
}
```