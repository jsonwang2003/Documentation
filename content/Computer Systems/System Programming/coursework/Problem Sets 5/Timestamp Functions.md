## What is a Unix timestamp?
A Unix timestamp is the number of seconds that have elapsed since the Unix epoch: 1970-01-01 00:00:00 UTC. It is an integer value (commonly stored in C as `time_t`) that represents an absolute point in time independent of any time-zone. Note that typical Unix timestamps count whole seconds (not milliseconds) and POSIX time does not account for leap seconds.

---
## Task
Implement two functions for working with timestamps in the chat server.
### Function 1: `get_current_timestamp`
- Returns the current Unix timestamp (seconds since epoch)
#### Function Signature
```c
// Returns the current time as a Unix timestamp
time_t get_current_timestamp(void);
```
### Function 2: `format_timestamp`
- Takes a Unix timestamp and formats it as a string in the format `YYYY-MM-DD HH:MM`
- Writes the formatted string to the provided buffer
#### Function Signature
```c
// Format a timestamp as "YYYY-MM-DD HH:MM" and write to buffer 
void format_timestamp(time_t timestamp, char* buffer);
```
### Timestamp Format
The timestamp format should be:

```
YYYY-MM-DD HH:MM
```

Where:
- `YYYY` is the four-digit year, like `2024`
- `MM` is the two-digit month (01-12), like `10` for October
- `DD` is the two-digit day (01-31), like `06`
- `HH` is the two-digit hour in 24-hour format (00-23), like `09` for 9am or `14` for 2pm
- `MM` is the two-digit minute (00-59), like `01` or `45`

All values should be zero-padded. For example:
- `2024-10-06 09:01` (correct)
- `2024-10-6 9:1` (incorrect - missing zero padding)
### Note
- For `get_current_timestamp`, use the `time()` system call from `<time.h>`
- For `format_timestamp`, use `localtime()` and `strftime()` from `<time.h>`
- The `strftime()` function with format string `"%Y-%m-%d %H:%M"` produces the exact format needed
- The buffer for `format_timestamp` will always be large enough (at least 17 bytes)
- Include `<time.h>` for the time functions

---
## Example
| Unix Timestamp | Formatted Output   |
| -------------- | ------------------ |
| 1728219660     | `2024-10-06 09:01` |
| 1728219720     | `2024-10-06 09:02` |
| 1728219840     | `2024-10-06 09:04` |
| 0              | `1970-01-01 00:00` |
| 1609459200     | `2021-01-01 00:00` |

---
## Code
```c
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

time_t get_current_timestamp(void)
{
    return time(NULL);
}

void format_timestamp(time_t timestamp, char *buffer)
{
    struct tm* timeinfo = localtime(&timestamp);
    strftime(buffer, 32, "%Y-%m-%d %H:%M", timeinfo);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <timestamp>\n", argv[0]);
        return 1;
    }

    // Test format_timestamp with provided timestamp
    time_t t = (time_t)atoll(argv[1]);
    char buffer[32];
    format_timestamp(t, buffer);
    printf("%s\n", buffer);

    // Optionally test get_current_timestamp
    // time_t current = get_current_timestamp();
    // printf("Current timestamp: %ld\n", current);

    return 0;
}
```