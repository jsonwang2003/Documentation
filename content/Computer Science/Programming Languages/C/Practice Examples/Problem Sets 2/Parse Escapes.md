## Tasks
Implement `parse_escapes`
- Parse **escape sequences** in a string and **convert** them to their **actual character representation**
- The function should process `\n`, `\t`, `\\`, and `\"` escape sequences **in-place**
- The input strings contain the literal characters `\` and `n` (for an actual newline), and need to convert them to actual newlines, tabs, etc.
### Function Signature
```c
// Parse escape sequences in a string in-place.
// Converts:
//   \n -> newline character
//   \t -> tab character
//   \\ -> backslash character
//   \" -> quote character
void parse_escapes(char str[]);
```
---
## Test
```bash
$ gcc parse_escapes.c -o parse_escapes
$ ./parse_escapes 
hello\nworld
hello
world
test\ttab
test	tab
quote\"here
quote"here
backslash\\test
backslash\test
simple
simple
$ ./parse_escapes < small_input.txt
hello
world
test	tab
quote"here
backslash\test
simple
```
---
## Code
```c
void parse_escapes(char str[]){
	for(int i = 0; str[i] != 0; i++){
		if(str[i] == '\\'){
			if(str[i+1] == 'n'){
                str[i] = '\n';
            } else if(str[i+1] == 't'){
                str[i] = '\t';
            } else if(str[i+1] == '"'){
                str[i] = '"';
            } else if(str[i+1] == '\\'){
                str[i] = '\\';
            }
            for(int j = i + 1; str[j] != 0; j++){
                str[j] = str[j + 1];
            }
		}
	}
}
```