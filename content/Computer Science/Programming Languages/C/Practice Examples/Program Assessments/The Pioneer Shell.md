## Learning Goals
This assignment calls upon many of the concepts that you have practiced in previous PAs. Specifically, we will practice the following concepts in C:

- string manipulation using library functions,
- command line arguments,
- process management using fork(), exec(), and wait(), and of course,
- using the terminal and vim.

We'll also get practice with a new set of library functions for opening, reading from, and writing to files.

## Introduction
Throughout this quarter, you have been interacting with the ieng6 server via the terminal – you've used `vim` to write code, used `gcc` and `make` to compile, used `git` to commit and push your changes, etc. At the heart of all this lies the shell, which acts as the user interface for the operating system.

At its core, the shell is a program that reads and parses user input, and runs built-in commands (such as `cd`) or executable programs (such as `ls`, `gcc`, `make`, or `vim`).

As a way to wrap up this quarter, you will now create your own shell (a massively simplified one of course). We shall call it– (drumroll please)
### The Pioneer Shell

The **PIoneer SHell**, or as we endearingly call it, `pish` (a name with such elegance as other popular programs in the UNIX world, e.g., `git`).

## Overall Task
### Interactive Mode
Your basic shell simply runs on an infinite loop; it repeatedly:

- prints out a prompt,
- reads keyboard input from the user
- parses the input into a command and argument list
- if the input is a built-in shell command, then executes the command directly
- otherwise, creates a child process to run the program specified in the input command, and waits for that process to finish

This mode of operation is called the **interactive mode**.
### Batch Mode
Shell programs (like `bash`, which is what you have been using on ieng6) also support a batch execution mode, i.e., scripts. In this mode, instead of printing out a prompt and waiting for user input, the shell reads from a script file and executes the commands from that file one line at a time.

In both modes, once the shell hits the end-of-file marker (EOF), it should call `exit(EXIT_SUCCESS)` to exit gracefully. In the interactive mode, this can be done by pressing `Ctrl-D`.

## Parsing Input
Every time the shell reads a line of input (be it from stdin or from a file), it breaks it down into our familiar `argv` array.

For instance, if the user enters `"ls -a -l\n"` (notice the newline character), the shell should break it down into `argv[0] = "ls"`, `argv[1] = "-a"`, and `argv[2] = "-l"`. In the starter code, we provide a struct to hold the parsed command.
### Handling Whitespaces
You should make sure your code is robust enough to handle various sorts of whitespace characters. In this PA, we expect your shell to handle any arbitrary number of spaces ( ) and tabs (\t) between arguments.

For example, your shell should be able to handle the following input: `" \tls\t\t-a -l "`, and still run the ls program with the correct `argv` array.

`strtok` will be **VERY** helpful for this.

## Built-In Commands
Whenever your shell executes a command, it should check whether the command is a built-in command or not. Specifically, the first whitespace-separated value in the user input string is the command. For example, if the user enters `ls -a -l tests/`, we break it down into `argv[0] = "ls"`, `argv[1] = "-a"`, `argv[2] = "-l"`, and `argv[3] = "tests/"`, and the command we are checking for is `argv[0]`, which is `"ls"`.

If the command is one of the following built-in commands, your shell should invoke your implementation of that built-in command.

There are three built-in commands to implement for this project: `exit`, `cd`, and `history`.

### Built-in Command: `exit`
When the user types `exit`, your shell should call the exit system call with `EXIT_SUCCESS` (macro for 0). This command does not take arguments. If any is provided, the shell raises a usage error.
### Built-in Command: `cd`
`cd` should be run with precisely 1 argument, which is the path to change to. You should use the `chdir()` system call with the argument supplied by the user. If `chdir()` fails (refer to man page to see how to detect failure), you should use call `perror("cd")` to print an error message.
### Built-in Command: `history`
When the user enters the `history` command, the shell should print out the list of commands a user has ever entered in interactive mode.

(If you are on ieng6, open the `~/.bash_history` file to take a look at all the commands you have executed. How far you've come this quarter!)

To do this, we will need to write the execution history to a file for persistent storage. Just like bash, we designate a hidden file in the user's home directory to store the command history.

Our history file will be stored at `~/.pish_history`. (You will find a function in the starter code that will help you get this file path.)

**Important:** When adding a command to history, if the user enters an empty command (0 length or whitespace-only), it should not be added to the history.

When the user types in the `history` command to our shell, it should print out all the contents of our history file, adding a counter to each line:

```bash
▶ history
1 history
▶ pwd
/home/jpolitz/cse29fa25/pa3/Simple-Shell
▶ ls
Makefile  script.sh  pish  pish.c  pish.h  pish_history.c
pish_history.o  pish.o
▶ history
1 history
2 pwd
3 ls
4 history
```

> [!NOTE]
> the number before each line is added by `history`. The contents of `.pish_history` should not contain the leading numbers.

## Running Programs
If, instead, the command is not one of the aforementioned built-in commands, the shell treats it as a program, and spawns a child process to run and manage the program using the `fork()` and `exec()` family of system calls, along with `wait()`.

## Excluded Features
accustomed to using that will not be present in our shell. (Just so you are aware how much work the authors of the bash shell put into their product!)

You will not be able to:

- use the arrow keys to navigate your command history,
- use Tab to autocomplete commands,
- use the tilde character (~) to represent your home directory,
- use redirection (> and <),
- pipe between commands (|),
- and many more…

Don't be concerned when these things don't work in your shell implementation!

If this were an upper-division C course, we would also ask you to implement redirection and piping.

## Handling Errors
Because the shell is quite a complex program, we expect you to handle many different errors and print appropriate error messages. To make this simple, we now introduce:
### Usage Errors
This only applies to built-in commands. When the user invokes one of the shell's built-in commands, we need to check if they are doing it correctly.

- For `cd`, we expect `argc == 2`,
- For `history` and `exit`, we expect `argc == 1`.

If the user enters an incorrect command, e.g. `exit 1` or `cd` without a path, then you should call the `usage_error()` function in the starter code, and continue to the next iteration of the loop.
### Errors to Handle
You need to handle errors from the following system calls/library functions using `perror()`. Please pay attention to the string we give to `perror()` in each case and reproduce it in your code exactly.

- `fopen()` failure: `perror("open")`,
- `chdir()` failure: `perror("cd")`,
- `execvp()` failure: `perror("pish")`,
- `fork()` failure: `perror("fork")`

## The Code Base
You are given the following files:

- **pish.h**: Defines `struct pish_arg` for handling command parsing; declares functions handling the history feature.
- **pish.c**: Implements the shell, including parsing, some built-in commands, and running programs.
- **pish_history.c**: Implements the history feature.
- **Makefile**: Builds the project.
- **ref-pish**: A reference implementation of the shell. Note that in this version, the history is written to `~/.ref_pish_history` rather than `~/.pish_history`, to avoid conflict with your own shell program.

### `struct pish_arg`
In `pish.h`, you will find the definition of `struct pish_arg`:

```h
#define MAX_ARGC 64

struct pish_arg {
    int argc;
    char *argv[MAX_ARGC];
};
```

## Running pish
First, run `make` to compile everything. You should see the `pish` executable in your assignment directory.

To run pish in interactive mode (accepting keyboard input), type

```bash
$ ./pish
```

Or, to run a script (e.g., script.sh), type

```bash
$ ./pish script.sh
```

The same applies for the reference implementation `ref-pish`.

---
## Code
Repository found [here](https://github.com/ucsd-cse29-fa25/pa3-pish-jsonwang2003)
### `pish.c`
```c
#include <ctype.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pish.h"

/*
 * Batch mode flag. If set to 0, the shell reads from stdin. If set to 1,
 * the shell reads from a file from argv[1].
 */
static int script_mode = 0;

/*
 * Prints a prompt IF NOT in batch mode (see script_mode global flag),
 */ 
int prompt(void)
{
    static const char prompt[] = {0xe2, 0x96, 0xb6, ' ', ' ', '\0'};
    if (!script_mode) {
        printf("%s", prompt);
        fflush(stdout);
    }
    return 1;
}

/*
 * Print usage error for built-in commands.
 */
void usage_error(void)
{
    fprintf(stderr, "pish: Usage error\n");
    fflush(stdout);
}

/*
 * Break down a line of input by whitespace, and put the results into
 * a struct pish_arg to be used by other functions.
 *
 * @param command   A char buffer containing the input command
 * @param arg       Broken down args will be stored here.
 */
void parse_command(char *command, struct pish_arg *arg)
{
	int i = 0;
	char* current = strtok(command, " \t\n");
	while(current != NULL){
		arg->argv[i] = current;
		i++;
		current = strtok(NULL, " \t\n");
	}
	arg->argv[i] = NULL;
	arg->argc = i;
	current = NULL;
}

/*
 * Run a command. 
 *
 * Built-in commands are handled internally by the pish program.
 * Otherwise, use fork/exec to create child process to run the program.
 *
 * If the command is empty, do nothing.
 * If NOT in batch mode, add the command to history file.
 */
void run(struct pish_arg *arg)
{
	if(arg->argc == 0){return;}

	char* command = arg->argv[0];
	if(script_mode == 0){
		add_history(arg);
	}
	if(strcmp(command, "exit") == 0){
		if(arg->argc != 1){
			usage_error();
			return;
		}
		exit(EXIT_SUCCESS);
	}

	if (strcmp(command, "cd") == 0){
		if(arg->argc != 2){
			usage_error();
			return;
		}
		if(chdir(arg->argv[1]) < 0){
			perror("cd");
		}
		return;
	}

	if(strcmp(command, "history") == 0){
		print_history();
		return;
	}

	int pid = fork();
	if(pid < 0){
		perror("fork");
		exit(1);
	} else if(pid == 0){
		execvp(command, arg->argv);
		perror("pish");
		exit(1);
	} else {
		waitpid(pid, NULL, 0);
		return;
	}
}

/*
 * The main loop. Continuously reads input from a FILE pointer
 * (can be stdin or an actual file) until `exit` or EOF.
 */
int pish(FILE *fp)
{
    // assume input does not exceed buffer size
    char buf[1024];
    struct pish_arg arg;

	while(prompt() && fgets(buf, sizeof(buf), fp) != NULL){
		parse_command(buf, &arg);
		run(&arg);
	}

    return 0;
}

int main(int argc, char *argv[])
{
    FILE *fp;

	if(argc == 1){
		script_mode = 0;
		fp = stdin;
	} else if (argc == 2){
		script_mode = 1;
		fp = fopen(argv[1], "r");
	} else {
		usage_error();
		exit(1);
	}

    pish(fp);

	if(argc == 2){
		fclose(fp);
	}

    return EXIT_SUCCESS;
}
```
### `pish.h`
```h
#ifndef __PISH_HISTORY_H__
#define __PISH_HISTORY_H__

#define MAX_ARGC 64

/*
 * Each input command will be parsed into this struct.
 * E.g. 
 *   If input is "du -h -d 1", we will have
 *      argc    = 4
 *      argv[0] = "du"
 *      argv[1] = "-h"
 *      argv[2] = "-d"
 *      argv[3] = "1"
 *      argv[4] = NULL
 *   It is IMPORTANT that the array is terminated by a NULL element.
 * This is different from the strings being NULL-terminated. If there
 * are 4 elements as in the example above, then argv[0] thru argv[3]
 * should contain the individual arguments, but argv[4] should be NULL.
 * This makes it easy to use argv to call `execvp()`.
 */
struct pish_arg {
    int argc;               /* The number of arguments */
    char *argv[MAX_ARGC];   /* NULL-terminated array of argument strings */
};

void add_history(const struct pish_arg *arg);

void print_history();

#endif // __PISH_HISTORY_H__
```
### `pish_history.c`
```c
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "pish.h"

static char pish_history_path[1024] = {'\0'};

/*
 * Set history file path to ~/.pish_history.
 */
static void set_history_path()
{
    const char *home = getpwuid(getuid())->pw_dir;
    strncpy(pish_history_path, home, 1024);
    strcat(pish_history_path, "/.pish_history");
}

void add_history(const struct pish_arg *arg)
{
    // set history path if needed
    if (!(*pish_history_path)) {
        set_history_path();
    }
    
	FILE* file = fopen(pish_history_path, "a");
	if(file == NULL){
		perror("open");
		exit(1);
	}

	if(arg->argc == 0){
		return;
	}

	fprintf(file, "%s", arg->argv[0]);
	for(int i = 1; i < arg->argc; i++){
		fprintf(file, " %s", arg->argv[i]);
	}
	fprintf(file, "\n");
	fclose(file);
}

void print_history()
{
    // set history path if needed
    if (!(*pish_history_path)) {
        set_history_path();
    }

	FILE* file = fopen(pish_history_path, "r");

	char buffer[100];
	int i = 1;
	while(fgets(buffer, sizeof(buffer), file) != NULL){
		printf("%d %s", i++, buffer);
	}

	fclose(file);    
}
```