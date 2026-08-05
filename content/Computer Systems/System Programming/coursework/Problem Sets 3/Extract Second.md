## Task
Implement the function `extract_second`
- Returns the second string in a space-separated string. 
- If there is a second string, print out the string "No second string"
- Can use [[Split Space]] function as helper function
- Do not print anything inside of the `extract_second` function

### Function Signature
```c
// Write a function that returns the second string in a space-separated string. If there is not a second string, print out the string "No second string".
//
// For example:
//
// Input: Hello world, this is a string!
// Output: world,
//
// Input: OneWord
// Output: No second string
//
char* extract_second(char* s);
```
---
## Example
```bash
$ gcc extract_second.c -o extract_second
$ ./extract_second
Hello world, this is a string!
world,
OneWord
No second string
$ ./extract_second < small_input.txt
world
29
No second string
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

char* extract_second(char* s){
    char* withoutFirst = split_space(s);
    if (withoutFirst != s){
        split_space(withoutFirst);
        return withoutFirst;
    } else {
        return "No second string";
    }
}
```