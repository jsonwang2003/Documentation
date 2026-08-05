## Task
In the file `decode2.c` with a main function that reads a line from the user. It should **assume** there are at least two bytes of standard input, and that the first two bytes are a 2-byte UTF-8 encoded character. The program should print the code point of that character in decimal format.

1. Create your program in `decode2.c`
2. Then use your program to **calculate the code point for `"ɽ"` and save it in the file `codepoint.txt`**. You may want to copy-paste it into the terminal to give it as input to your program. Here it is on its own line to make copy-pasting easy: 
	- `ɽ`

---
## Example
```bash
$ gcc decode2.c -o decode2
$ ./decode2
ɒ      # this line is the typed input (probably copy-paste from the question)
594    # this line is the output you should save to codepoint.txt
```

---
## Code
```c
#include <stdio.h>

int main(int argc, char** argv){
    char buffer[3];
    fgets(buffer, sizeof(buffer), stdin);

    int codepoint = ((buffer[0] & 0x1F) << 6) + (buffer[1] & 0x3F);

    printf("%d\n", codepoint);
}
```