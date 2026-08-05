## Task
Implement the function `format_reaction_line`.
- Write a function that formats a single reaction according to the PA specification
- The format is: `(username) reaction` with some leading spaces
- Write the result into a provided buffer
### Function Signature
```c
struct Reaction {
	char user[16];
	char message[16];
};

// Format a reaction as "                        (username) reaction"
// buffer should have enough space for the formatted string
void format_reaction_line(struct Reaction* reaction, char* buffer);
```
### Notes
- Use `sprintf()` or `snprintf()` to format the string
- The format includes leading spaces (about 24 spaces to align with chat lines)
- Then `(username)` with no spaces inside the parentheses
- Then a space, then the reaction message
- According to the PA spec: "some space before the (, no space between the () and the username, and some space after the ) before the reaction message"
- You can use a format like: `" (%s) %s"` (24 leading spaces)
- Include `<stdio.h>` for `sprintf()`

---
## Example
|Reaction|Output|
|---|---|
|user="joe", message="👍"|`(joe) 👍`|
|user="aaron", message="cool"|`(aaron) cool`|
|user="alice", message="😬"|`(alice) 😬`|

---
## Code
```c
#include <stdio.h>
#include <string.h>

void format_reaction_line(char *user, char *reaction, char *buffer){
    sprintf(buffer, "(%s) %s", user, reaction);
}
```