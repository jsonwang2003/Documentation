## Task
Implement the function `split_space`
- Gets the first string in a space-separated string by *changing* the string to have a null terminator at the first space
- Return the pointer right after the null terminator
- No effect and returns the original pointer if no spaces

### Function Signature
```c
// Returns the first string in a space-separated string by changing the string 
// to have a null terminator at the first space, and returning the pointer right
// after the null terminator. Has no effect and returns the original pointer if
// no spaces. Do not print inside of the split_space function. this is done in main.
//
// For example:
//
// Input: Hello world, this is a string!
// Output:
// First string: Hello
// Split string: world, this is a string!
//
// Input: OneWord
// Output:
// First string: OneWord
// Split string: OneWord
//
char* split_space(char* s);
```
---
## Example
```bash
$ gcc split_space.c -o split_space
$ ./split_space
Hello
Original string: Hello
Split string: Hello
Hello world, this is a sentence!
Original string: Hello
Split string: world, this is a sentence!
$ ./split_space < small_input.txt
Original string: Hello
Split string: world
Original string: CSE
Split string: 29 is the best class!
Original string: OneWord
Split string: OneWord
```
---
## Code
```c
char* split_space(char* s){
	int i = 0;
	while(s[i] != 0){
		if(s[i] == ' '){
			s[i] = 0;
			return &s[i+1];
		}
		i++;
	}
	return s;
}
```