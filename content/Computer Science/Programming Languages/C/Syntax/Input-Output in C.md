## Standard Input-Output
Every running program begins with three default I/O streams
- `stdout`: Printing output to **terminal**
- `stdin`: read in input from **keyboard**
- `stderr`: Printing error to **terminal**
Can change the location that a running program's `stdin` , `stdout` and/or `stderr` read from or write to
#### 1. Redirecting to a file
```bash
# redirect a.out's stdin to read from file infile.txt: 
$ ./a.out < infile.txt 

# redirect a.out's stdout to print to file outfile.txt: 
$ ./a.out > outfile.txt 

# redirect a.out's stdout and stderr to a file out.txt 
$ ./a.out &> outfile.txt 

# redirect all three to different files: 
# (< redirects stdin, 1> stdout, and 2> stderr): 
$ ./a.out < infile.txt 1> outfile.txt 2> errorfile.txt
```

### `printf`
Resembles formatted print calls in [[Python/Format Print|Python]], where the caller specifies a format string to print.
When adding placeholders in a format string passed to `printf`, **pass their corresponding values as additional arguments following the format string**
```c
printf(s, arguments)
```
##### Format Strings
Often contains special format specifiers
- Special Characters
	- `\t`: print tabs
	- `\n`: prints newlines
- Placeholders for Values (`%` followed by a type specifier)
	- `%f`, `%g`: placeholders for a **float** or **double** value
	- `%d`: placeholder for a **decimal value** (char, short, int)
	- `%u`: placeholder for an **unsigned decimal** 
	- `%c`: placeholder for a **single character** 
	- `%s`: placeholder for a **string** value 
	- `%p`: placeholder to **print an address** value 
	- `%ld`: placeholder for a **long** value 
	- `%lu`: placeholder for an **unsigned long** value 
	- `%lld`: placeholder for a **long long** value 
	- `%llu`: placeholder for an **unsigned long long** value
##### Specifying the field width format placeholders
- `%5.3f`: print float value in space 5 chars wide, with 3 places beyond decimal 
- `%20s`: print the string value in a field of 20 chars wide, right justified 
- `%-20s`: print the string value in a field of 20 chars wide, left justified 
- `%8d`: print the int value in a field of 8 chars wide, right justified 
- `%-8d`: print the int value in a field of 8 chars wide, left justified
##### Defining placeholders for displaying values in different representations
- `%x`: print value in **hexadecimal** (*base 16*)
- `%o`: print value in **octal** (*base 8*)
- `%d`: print value in **signed decimal** (*base 10*)
- `%u`: print value in **unsigned decimal** (*base 10*)
- `%e`: print **float** or **double** in **scientific notation**
>[!NOTE]
>No formatting option to display a value in binary

### `scanf`
Provides one method for reading in values from `stdin` (usually from the user entering them via the keyboard) and storing them in program variables. 
**Picky on the format** in which the user enters the data
1. Takes a **format string** that specifies the **number** and **type** of input values to read in
2. *locations* of program variables into which the values should be stored
	- Programs typically **combine the *address of*** (`&`) operator with a variable name to produce the location of the variable in the program's memory → the memory address of the variable
```c
int x;
float pi;

// read in an int value followed by a float value ("%d%g")
// store the int value at hte memory location of x (&x)
// stroe the float value at the memory location of pi (&pi)
scanf("%d%g", &x, &pi);
```

> [!IMPORTANT]
> Individual input values must be separated by **at least one** whitespace character (spaces, tabs, newlines)
> **HOWEVER** `scanf` skips over **leading** and **trailing** whitespace characters as it find the start and end of each numeric literal value

Programmers often write format strings for `scanf` that only consist of placeholder specifiers without any other characters between them
### `getchar` and `putchar`
- `getchar`: read a single character value from `stdin`
	- particularly useful in C programs that **need to support careful error detection and handling of badly formed user input**
	- `scanf` is not robust in this way
- `putchar`: writes a single character value to `stdout`
```c 
ch = getchar();     // read in the next char value from stdin
putchar(ch);        // write the value of ch to stdout
```
## File Input-Output
The C standard library `stdio.h` includes a **stream interface** for file I/O
A **file** stores persistent data that lives **beyond the execution of the program that created it**
- Text file (`.txt`): represents a **stream of characters**, and each open file tracks its current position in the character stream.
- Text files may contain **special chars** like the `stdin` and `stdout` streams:
	- newlines `'\n'`
	- tabs `'\t'`

When opening a file, the **current** position starts at the very first character in the file, and it moves as a result of every character read (or written) to the file.
→ to read the **10th** character in a file, the **first 9 characters need to first be read** (or the current position must be explicitly moved to the 10th character using the `fseek` function)

### C's file interface
Views file as an input / output stream, and library functions read from or write to the next position in the file stream
- `fprintf` → `printf` counter part for files
- `fscanf` → `scanf` counter part for files
They use **format string** to specify what to write or read, and they includes arguments that provide values or storage for the data that gets written or read.

Library also provides:
- `fputc`: puts character into file
- `fgetc`: gets character from file
- `fputs`: puts argument value into file
- `fgets`: gets argument type from file

C's I/O library generates a **special end of file** character (`EOF`) that represents the end of the file.
Functions reading from a file can test of `EOF` to determine when they have reached the end of file stream

## Using Text Files in C
To read or write a file in C:
1. Declare a `FILE *` Variable
	```c
	FILE *infile;
	FILE *outfile;
	```
	- These declarations create **pointer variables** to a library-defined `FILE *` type
	- These pointers **cannot be dereferenced** in an application program → instead, they refer to **a specific file stream** when passed to I/O library functions
2. Open the File
	- Associate the variable with an actual file stream by calling `fopen`
	- When opening a file, the **mode parameter** determines whether the program opens it for:
		- reading ("r")
		- writing ("w")
		- appending ("a")
	```c
	// relative path name of file, read mode
	infile = fopen("input.txt", "r"); 
	if (infile == NULL){
		printf("Error: unable to open file %s\n", "input.txt");
		exit(1);
	}
	
	// fopen with absolute path name of the file, write mode
	outfile = fopen("/home/me/output.txt", "w");
	if(outfile = NULL){
		printf("Error: unable to open outfile\n");
		exit(1);
	}
	```
	- The `fopen` returns `NULL` to report errors, which may occur if **its given an invalid filename** or **user doesn't have permission to open the specified file**
3. Use I/O operations to read, write, or move the current position in the file:
	```c
	int ch; // EOF is not a char value, but is an int
	        // since all char values can be stored in int, use int for ch
	ch = getc(infile);        // read next char from infile stream
	if (ch != EOF){
		putc(ch, outfile);    // write value to the outfile
	}
	```
4. Close the file
	- use `fclose` to close the file when the program no longer needs it
	```c
	fclose(infile);
	fclose(outfile);
	```

The `stdio` library also provides functions to change the current position in a file
```c
// to reset current position to beginning of file
void rewind(FILE *f);

rewind(infile);

// to move to a specific location in the file
fseek(FILE *f, long offset, int whence);

fseek(f, 0, SEEK_SET); // seek to the beginning of the file 
fseek(f, 3, SEEK_CUR); // seek 3 chars forward from the current position
fseek(f, -3, SEEK_END); // seek 3 chars back from the end of the file
```
## Standard and File Input-Output in `stdio.h`
The C `stdio.h` library has many functions for reading and writing to files and to the standard file-like streams (`stdin`, `stdout`, `stderr`)
### Character Based
```c
// Returns the next character in the file stream (EOF is an int value)
int fgetc(FILE *f);

// Writes the char value c to the file stream f
// Returns the char value written
int fputc(int c, FILE *f);

// Pushes the character c back onto the file stream 
// At most one char (and not EOF) can be pushed back
int ungetc(int c, FILE *f);

// Like fgetc and fputc but for stdin and stdout
int getchar();
int putchar();
```

### String Based
```c
// Reads at most n-1 characters into the array s stopping if a newline is 
// encountered, newline is included in the array which is '\0' terminated
char *fgets(cahr *s, int n, FILE *f);

// Writes the string s (make sure '\0' terminated) to the file stream f
int fputs(cahr *s, FILE *f);
```

### Formatted
```c
// Writes the contents of the format string to file stream f
// (with placeholders filled in with subsequent argument values)
// returns the numebr of characters printed
int fprintf(FILE *f, char *format, ...);

// Like fprintf but to stdout
int printf(char *format, ...);

// Use fprintf to print stderr
int ret = 1;
fprintf(stderr, "Error return value: %d\n", ret);

// Read values specified in the format string from file stream f
//    Store the read-in values to program storage locations of types
//    matching the format string
// Returns number of input items converted and assigned
//    or EOF on error or if EOF was reached
int fscanf(FILE *f, char *format, ...);

// Like fscanf but reads from stdin
int scanf(char *format, ...);
```

> [!NOTE]
> In general, `scanf` and `fscanf` are sensitive to badly formed input. However, for file I/O, often programmers can **assume that an input file is well formatted**

The format string for `fscanf` can include the following syntax specifying types of values and ways of reading from the file system:
- `%d`: integer
- `%f`: float
- `%lf`: double
- `%c`: character
- `%s`: string, up to **first white space**
- `%[...]`: string up to **first character *not* in brackets**
- `%[0123456789]`: read in digits
- `%[^...]`: string, up to **first character in brackets**
- `%[^\n]`: read everything **up to a newline**

>[!NOTE]
>Can be tricky to get `fscanf` format string correct, particularly when **reading a mix of numeric and string / character types** from a file

```c
int x;
double d;
char c, array[MAX];

// Write int & char values to file separated by color with newline at the end
fprintf(outfile, "%d:%c\n", x, c);

// Read an int & char from file where int and char are separated by comma
fscanf(infile, "%d,%c", &x, &c);

// Read a string from a file into array (stops reading at whitespace char)
fscanf(infile, "%s", array);

// Read a double and a string up to 24 characters from infile
fscanf(infile, "%lf %24s", &d, array);

// Read in a string consisting of only char values in the specified set (0-5)
// stops reading when ...
//    20 chars have been read OR
//    a character not in the set is reached OR
//    the file stream reaches end-of-file (EOF)
fscanf(infile, "%20[012345]", array);

// Read in a string; stop when reaching a punctuation mark from the set
fscanf(infile, "%[^.,:!;]", array);

// Read in two integer values: store first in long, second in int
// Then read in a char value following the int value
fscanf(infile, "%ld %d%c", &x, &b, &c);
```

> [!NOTE]
> The final example above
> ```c
> fscanf(infile, "%ld %d%c", &x, &b, &c);
> ```
> The format string **explicitly reads in a character value after a number to ensure that the file stream's current position gets properly advanced for any subsequent calls to `fscanf`**
> 
> Example: this pattern is often used to **explicitly read in (and discard) a whitespace character (like '\n')** to ensure that **the next call to `fscanf` begins from the next line in the file**

> [!IMPORTANT]
> Reading an additional character is **necessary** if the *next* call to `fscanf` attempts to **read in a character value**


