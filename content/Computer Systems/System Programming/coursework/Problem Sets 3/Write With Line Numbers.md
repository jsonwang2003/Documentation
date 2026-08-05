## Task
Implement the function `write_with_line_numbers`
- Given a file path and a string, write the string out to the file **with line numbers at the beginning of each line**
- So a string like `"joe: hello\nolivia: yo\njoey: sup"` would be written as
```txt
1 joe: hello
2 olivia: yo
3 joey: sup
```

> [!IMPORTANT]
> The input strings to test with have a slash and n `('\n')`, not an actual newline character.
> Do not have to process the string literal `('\n')` in function, this is processed for you in `convert_escaped_newlines`. In your function `write_with_line_numbers`, treat the `'\n'` from the input string as an actual newline char

- Do not print anything inside of the `write_with_line_numbers` function, this is handled in the `main` function
### Function Signature
```c
// Given a file path and a string, write the string out to the file with line numbers
// at the beginning of each line
//
// For example:
// Input: "output.txt joe: hello\nolivia: yo\njoey: sup"
//        (path = "output.txt", to_save = "joe: hello\nolivia: yo\njoey: sup")
//
// Output (in output.txt):
// 1 joe: hello  
// 2 olivia: yo  
// 3 joey: sup
//
//
void write_with_line_numbers(char* path, char* to_save)
```

---
## Example
```bash
$ gcc write_with_line_numbers.c -o write_with_line_numbers
$ ./write_with_line_numbers
output.txt joe: hello\nolivia: yo\njoey: sup
$ cat output.txt 
1 joe: hello
2 olivia: yo
3 joey: sup
$ ./write_with_line_numbers < input.txt (saves result to result.txt)
$ cat result.txt 
1 A very long line of text. A very long line of text. A very long line of text.
2 Another long one here
3 And a third long one
4 More lines to test this function correctly
5 This is line number five without numbering in the text
6 Sixth line continues the example nicely
7 Seventh line here with more words to fill the space
8 Eighth line is also quite verbose and descriptive
9 Ninth line is getting close to the end
10 Tenth line finishes the first half nicely
```

---
## Code
```c

void write_with_line_numbers(char* path, char* to_save){
    FILE* file = fopen(path, "w");

    int line = 1, i, current;
    char buffer[100];
    for(i = 0, current = 0; to_save[i] != 0; i++){
        if(to_save[i] == '\n'){
            buffer[current] = 0;
            fprintf(file, "%d %s\n", line, buffer);
            line++;
            current = 0;
        } else {
            buffer[current] = to_save[i];
            current++;
        }
    }

    fprintf(file, "%d %s\n", line, buffer);
    fclose(file);
}
```