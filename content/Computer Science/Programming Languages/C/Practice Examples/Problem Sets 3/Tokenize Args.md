## Task
- Read delimiter characters from `argv` (all arguments after the program name)
- Read lines from `stdin`
- For each line, **tokenize** using **ALL** the delimiter characters provided
- Print each token with its index

**Command Format**: `./tokenize_args <delim1> <delim2> ...`
**Example**: `./tokenize_args a | /` → split by 'a', '|', and '/'

### Function Signature
```c
// Tokenize str using the delimiters in delim_str.
// Print each token with its index.
void tokenize_args(const char* str, const char* delim_str);
```

### `strtok`
A function used to tokenize a string based on a set of delimiter characters

`char* strtok(char* str, const char* delim);`

The `strtok()` breaks a string into a sequence of zero or more nonempty tokens. On the first call to `strtok()`, the string to be parsed should be specified in `str`. In each subsequent call that should parse same string, `str` must be `NULL`

The `delim` argument specifies a set of characters that delimit the tokens in the parsed string. Each call to `strtok()` returns a pointer to a null-terminated string containing the next token. If no more tokens are found, `strtok()` returns `NULL`

> [!TIP]
> Build a delimiter string from `argv`, then use `strtok(str, delim_str)`

#### Example Use of `strtok`
```c
char str[] = "hello world|test";
char *delim = " |";
char *token = strtok(str, delim);
int i = 0;
while (token != NULL) {
    printf("%d: %s\n", i++, token);
    token = strtok(NULL, delim);
}
```

---
## Example
```bash
$ gcc tokenize_args.c -o tokenize_args
$ ./tokenize_args " " "|"
hello world|test
0: hello
1: world
2: test
$ ./tokenize_args "a" "/" "|"
banana/apple|apricot
0: b
1: n
2: n
3: pple
4: pricot
```

---
## Code
```c
#include <string>

void tokenize_args(const char *str, const char *delim_str)
{
    char buffer[strlen(str) + 1];
    strcpy(buffer, str);

    int index = 0;
    char* word = strtok(buffer, delim_str);
    while(word != NULL){
        printf("%d: %s\n", index, word);
        index++;
        word = strtok(NULL, delim_str);
    }
}
```