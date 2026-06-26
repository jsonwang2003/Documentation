## Task
Implement the function `find_in_file`
- Given a file path and a string to find, return `1` (true) if that string is in the file, `0` (false) otherwise.
- Also return `0` (false) if the file does not exist or there is any other file access error
- Files to test with are located in the `files` directory
- Do not print anything inside of the `find_in_file` function, this is handled in the `main` function.
### Function Signature
```c
// Given a file path and a string to find, return 1 (true) if that string is in 
// the file, 0 (false) otherwise. Also return 0 (false) if the file does not
// exist or there is any other file access error
//
// For example:
//
// Input: files/file1.txt hello world (substring to find is "hello world")
// Output: 1
//
// Input: files/file2.txt asdfjkl;
// Output: 0
//
// Input: files/doesnotexist.txt hello!
// Output: 0
//
//
uint8_t find_in_file(char* path, char* to_find);
```

---
## Example
```bash
$ gcc find_in_file.c -o find_in_file
$ ./find_in_file
files/file1.txt hello world
1
files/file2.txt asdfjkl;
0
files/doesnotexist.txt hello!
0
$ ./find_in_file < small_input.txt
1
0
0
```

---
## Code
```c
#include <stdint.h>
#include <string.h>
#include <stdio.h>

uint8_t find_in_file(char* path, char* to_find) {
    FILE* file = fopen(path, "r");

    if(file == NULL){
        return 0;
    }

    char buffer[1000];
    while(fgets(buffer, sizeof(buffer), file)!= NULL){
        for(int front = strlen(to_find) - 1, back = 0; buffer[front] != 0; front++, back++){
            char current[strlen(to_find) + 1];
            for(int i = 0; i < strlen(to_find); i++){
                current[i] = buffer[back + i];
            }
            current[strlen(to_find) ] = 0;

            if(strcmp(current, to_find) == 0) {
                fclose(file);
                return 1;
            }
        }
    }

    fclose(file);
    return 0;
}
```