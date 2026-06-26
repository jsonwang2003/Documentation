## Task
Implement the function `format_chat_line`.
- Write a function that formats a single chat (without reactions) according to the PA specification
- The format is: `[#N YYYY-MM-DD HH:MM] username: message`
- Write the result into a provided buffer
### Function Signature
```c
// Format a chat as "[#N YYYY-MM-DD HH:MM] username: message"
// buffer should have enough space for the formatted string
void format_chat_line(uint32_t id, char *user, char *message, char *timestamp,
	uint32_t num_reactions, char *buffer);
```

### Note
- Use `sprintf()` or `snprintf()` to format the string
- The format is: `[#<id> <timestamp>] <user>: <message>`
- There should be at least one space between the ID and timestamp
- There should be at least one space between the `]` and username
- There should be no space after the username before the `:`
- There should be one space after the `:` before the message
- You don't need to handle alignment/padding (just basic spacing)
- Make sure the buffer is large enough (at least 300+ bytes to be safe)
- Include `<stdio.h>` for `sprintf()`

---
## Example
|Chat|Output|
|---|---|
|id=1, user="joe", timestamp="2024-10-06 09:01", message="hi aaron"|`[#1 2024-10-06 09:01] joe: hi aaron`|
|id=5, user="aaron", timestamp="2024-10-06 09:02", message="sup"|`[#5 2024-10-06 09:02] aaron: sup`|
|id=123, user="kruti", timestamp="2024-10-24 13:01", message="hello world"|`[#123 2024-10-24 13:01] kruti: hello world`|

---
## Code
```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void format_chat_line(uint32_t id, char *user, char *message, char *timestamp,
                      uint32_t num_reactions, char *buffer)
{
    sprintf(buffer, "[#%u %s] %s: %s", id, timestamp, user, message);
}
```