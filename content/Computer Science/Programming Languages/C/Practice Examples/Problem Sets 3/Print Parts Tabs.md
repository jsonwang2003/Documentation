## Task
Implement the function `print_parts_tabs`
- Print each tab-separated part of the string, one per line, in the format `index: part`
- Consecutive tabs should be treated as a single separator
- Ignore leading and trailing tabs
### Function Signature
```c
// Print each tab-separated part of str, one per line, labeled with its index.
// Format: "index: part"
void print_parts_tabs(const char* str);
```

---
## Example
```bash
$ gcc print_parts_tabs.c -o print_parts_tabs
$ ./print_parts_tabs
ls	-l	./
0: ls
1: -l
2: ./
	hello		world	this		is		c	
0: hello
1: world
2: this
3: is
4: c
$ ./print_parts_tabs < small_input.txt
0: ls
1: -l
2: ./
0: hello
1: world
2: this
3: is
4: c
0: one
1: two
1: three
$ # The next command is how you should create the output files
$ # It will result in a new file with the output from running ./print_parts_tabs, which
$ # the grader will check for
$ ./print_parts_tabs < small_input.txt > small_result.txt
$ ./print_parts_tabs < input.txt > result.txt
```

---
## Code
```c
void print_parts_tabs(const char *str){
    int index = 0, i = 0, current = 0;
    char word[100];
    while(str[i] == '\t'){i++;}

    while(str[i] != 0){
        if(str[i] != '\t'){
            word[current] = str[i];
            current++;
            i++;
        } else {
            word[current] = 0;
            if(current > 0){
                printf("%d: %s\n", index, word);
                index++;
            }
            current = 0;
            i++;
        }
    }

    if(current > 0){
        word[current] = 0;
        printf("%d: %s\n", index, word);
    }
}
```