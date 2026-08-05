## Task
- Implement a program that accepts command-line flags:
	- `-d <delim>` → use custom delimiter string (default: space)
	- `-s <sep>` → use custom output separator (default: newline)
- Read lines from `stdin`, tokenize by delimiter, output tokens separated by separator
- Flags can appear in any order

**Command Format**: `./parse_flags [-d <delim>] [-s <sep>]`

> [!IMPORTANT]
> Process flags before reading `stdin`. After processing all flags, read from `stdin`

> [!TIP]
> - Use `strtok()` with the delimiter string
> - Parse `argv` to find flags and their values
> - Default delimiter: `" "` (space)
> - Default separator: `"\n"` (newline)
> - Print separator **BETWEEN** tokens, not after the last one

---
## Example
```bash
$ gcc parse_flags.c -o parse_flags
$ ./parse_flags
hello world test
hello
world
test
$ ./parse_flags -d ":"
key:value:pair
key
value
pair
$ ./parse_flags -s "\t"
hello world test
hello	world	test
$ ./parse_flags -d ":" -s " | "
a:b:c
a | b | c
$ ./parse_flags -s "," -d "|"
one|two|three
one,two,three
```

---
## Code
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    char *delim = " "; // default delimiter
    char *sep = "\n";  // default separator

    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-d") == 0){
            i++;
            delim = argv[i];
        }
        if(strcmp(argv[i], "-s") == 0){
            i++;
            sep = argv[i];
        }
    }

    // After parsing flags, read from stdin
    char buffer[1001];

    while (fgets(buffer, sizeof(buffer), stdin) != NULL)
    {
        // Remove trailing newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
        {
            buffer[len - 1] = '\0';
        }

        char* word = strtok(buffer, delim);
        while(word != NULL){
            printf("%s", word);
            word = strtok(NULL, delim);
            if (word != NULL) {
                printf("%s", sep);
            }
        }
    }

    return 0;
}
```