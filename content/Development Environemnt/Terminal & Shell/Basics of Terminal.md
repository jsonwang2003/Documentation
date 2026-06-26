## Terminal Basics
The terminal is a text-based interface that interprets commands and outputs results.
### Navigation and Discovery
- **echo**: Displays the string provided to it.
    - Example: `echo "Hello World"` outputs `Hello World`.
- **pwd** (Print Working Directory): Prints the absolute path of your current location.
    - Example: `pwd` → `C:/Users/jason`.
- **ls**: Lists the files and folders contained within the current directory.
### Previewing Files
- **head**: Displays the beginning of a file.
    - Default (10 lines): `head orders.tsv`
    - Custom count: `head -n 3 orders.tsv` (Prints first 3 lines)
- **tail**: Displays the end of a file.
    - Default (10 lines): `tail orders.tsv`
    - Custom count: `tail -n 3 orders.tsv` (Prints last 3 lines)
- **cat**: Returns and displays the entire content of a file.

---
## Data Filtering and Pipelines
Unix commands can be combined to filter data and store results.
### Searching with grep
The `grep` command generates lists based on filtering properties.
- **Standard**: `grep "Burrito" orders.tsv` (Case-sensitive search).
- **Case-Insensitive**: `grep -i "burrito" orders.tsv` (Ignores casing).
- **Invert Search**: `grep -v "Burrito" orders.tsv` (Finds lines without the term).
### Connecting Commands
- **Pipes (`|`)**: Directs the output of the left command to the input of the right command.
    - Example: `grep -v "Burrito" orders.tsv | tail -n 3`
- **Redirection (`>`)**: Saves the result of an analysis into a specific file.
    - Example: `grep "Burrito" orders.tsv > burritos.tsv`
- **File Copying (`cp`)**: Copies content from one file to another.
    - Example: `cp ref.fa.bak ref.fa`

---
## Environment Variables

Variables store data for use throughout the terminal session.
- **Assignment**: Use `=` with no spaces around it.
    - Example: `abc=123`
- **Retrieval**: Use the `$` delimiter to call the variable.
    - Example: `echo $abc`
- **Deletion**: Use the `unset` command.
    - Example: `unset abc`
- **Listing**: Use `env` to see all active environment variables.

---
## Compiling C++ Programs
C++ source code is converted into an executable using the `g++` compiler.
### Standard Compilation
For a file named `HelloWorld.cpp`:
1. **Compile**: `g++ HelloWorld.cpp` (Creates a default file named `a.out`).
2. **Execute**: `./a.out`.
3. **Named Output**: Use the `-o` flag to specify the executable name.
    - `g++ -o hello_world HelloWorld.cpp`
    - `./hello_world`
### Build Automation with Makefiles
Makefiles organize the compilation of complex projects with multiple files. They track modifications and only recompile necessary components.

**General Syntax:**
```makefile
target: dependencies
	system command
```

> [!Note]
> System commands must be indented with a Tab character.

**Example: Multiple Dependencies** If `HelloWorld.cpp` depends on `PrintWords.cpp`:

```
all: hello_world

hello_world: HelloWorld.o PrintWords.o
	g++ -o hello_world HelloWorld.o PrintWords.o
	
HelloWorld.o: HelloWorld.cpp
	g++ -c HelloWorld.cpp
	
PrintWords.o: PrintWords.cpp
	g++ -c PrintWords.cpp
	
clean:
	rm *.o hello_world
```

**Using the Makefile:**
1. **Build**: Run `make` (Executes the first target, `all`).
2. **Execute**: Run `./hello_world`.
3. **Maintenance**: Run `make clean` to remove compiled object files and binaries.