## Task
Implement the function `get_line`
- Given a **file path** and a **line number**
	- write the first **max bytes** of the string (assumes result is at least max +1 bytes long)
	- Return the number of bytes written, or `-1` if the line number is invalid, or there is a file access error
- Do not print anything inside of the `get_line` function, this is handled in the `main` function

### Function Signature
```c
// Given a file path and a line number, write the first max bytes of the string on 
// that line into the result array, null terminating the string (assumes result is at 
// least max + 1 bytes long). Return the number of bytes written, or -1 if the line 
// number is invalid or there is a file access error
//
// For example:
//
// Input: files/file1.txt 1 15 (line 1, 15 bytes)
// Output: 14 cat cse29. (14 is the number of bytes written, "cat cse29.txt" is what's in result)
//
// Input: files/file2.txt 10000 20;
// Output: -1 (line 10000 doesn't exist)
//
// Input: files/nonexistent_file.txt 33 41
// Output: -1 (nonexistent_file doesn't exist)

// files/file3.txt 62 112
//
//
int32_t get_line(char* path, int line_number, char result[], int max)
```

---
## Example
```bash
$ gcc get_line.c -o get_line
$ ./get_line
files/file3.txt 20 35
35 After the initial deadline, a resub
files/file4.txt 12 12 
-1
files/file1.txt 1000000 12
-1
files/file1.txt 13 100
5 exit
$ ./get_line < small_input.txt
14 cat cse29.txt
-1
-1
```

---
## Code
```c
#include <stdint.h>
#include <stdio.h>

int32_t get_line(char* path, int line_number, char result[], int max) {
    FILE* file = fopen(path, "r");
    if(file == NULL){
        return -1;
    }

    int line = 1;
    char buffer[1000];
    while(line <= line_number && fgets(buffer, sizeof(buffer), file)){
        if(line == line_number){
            int i;
            for(i = 0; i < max && buffer[i] != 0; i++){
                result[i] = buffer[i];
            }
            result[i] = 0;
            fclose(file);
            return i;
        }
        line++;
    }

    fclose(file);
    return -1;
}
```