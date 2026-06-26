> [!INFO] Source Attribution
> These notes are adapted from the **[GPROF Tutorial – How to use Linux GNU GCC Profiling Tool](https://www.thegeekstuff.com/2012/08/gprof-tutorial/)** by Himanshu Arora at *The Geek Stuff*.
### Introduction to `gprof`
Profiling identifies time-consuming parts of a program, allowing developers to optimize execution speed. **`gprof`** provides a flat profile (timing per function) and a call graph (relationship between functions).
#### Workflow
1. Compile with the `-pg` flag.
2. Execute the program to generate `gmon.out`.
3. Run the `gprof` tool to analyze the data.

---
## Steps to Use `gprof`
### Step 1: Compilation
You must enable profiling during both compilation and linking using the `-pg` flag.

```Bash
gcc -Wall -pg test_gprof.c test_gprof_new.c -o test_gprof
```

> [!Note]
> The option `-pg` can be used with `gcc` command that satisfies all of the following
> - `gcc` command that compiles (`-c` option)
> - `gcc` command that links(`-o` option on object files)
> - `gcc` command that does the both(as in example above).

### Step 2: Execution
Run the binary. It will produce a `gmon.out` file in the current working directory upon completion.

```Bash
./test_gprof
```
### Step 3: Analysis
Run `gprof` with the executable and the `gmon.out` file. Redirect the output to a text file for readability.

```bash
gprof test_gprof gmon.out > analysis.txt
```

---
## Understanding the Report
The output is divided into two main tables:
### Flat Profile

```txt
%    cumulative self          self   total
time seconds    seconds calls s/call s/call name
33.86 15.52     15.52    1    15.52  15.52  func2
33.82 31.02     15.50    1    15.50  15.50  new_func1
33.29 46.27     15.26    1    15.26  30.75  func1
0.07  46.30     0.03                        main
```

Shows the total time spent in each function and the call frequency.
- **% time:** The percentage of the total program runtime spent in this specific function.
- **cumulative seconds:** A running total of the time spent in this function plus all functions listed above it.
- **self seconds:** The actual time spent executing code within this function alone (the primary sorting metric).
- **calls:** The total number of times the function was executed during the profile run.
- **self s/call:** The average amount of time (in seconds or milliseconds) spent in the function per individual call.
- **total s/call:** The average time spent in the function and all its children/descendants per individual call.
- **name:** The unique identifier or name of the function being profiled.
### Call Graph

```txt
granularity: each sample hit covers 2 byte(s) for 0.02% of 46.30 seconds

index % time self children called name

[1]   100.0  0.03  46.27          main [1]
             15.26 15.50    1/1      func1 [2]
             15.52 0.00     1/1      func2 [3]
-----------------------------------------------
             15.26 15.50    1/1      main [1]
[2]   66.4   15.26 15.50    1     func1 [2]
             15.50 0.00     1/1      new_func1 [4]
-----------------------------------------------
             15.52 0.00     1/1      main [1]
[3]   33.5   15.52 0.00     1     func2 [3]
-----------------------------------------------
             15.50 0.00     1/1      func1 [2]
[4] 33.5     15.50 0.00     1     new_func1 [4]
-----------------------------------------------
```

#### Primary Function Entry
- **index:** A unique numeric ID used to cross-reference functions within the table.
- **% time:** The percentage of the total program runtime spent in this function and all of its children.
- **self:** The total time spent executing code strictly inside this function.
- **children:** The total time spent in all descendant functions called by this one.
- **called:** The number of non-recursive calls made to this function (recursive calls follow a `+`).
- **name:** The name of the function and its index number.
#### Parent Lines (Above the Function)
- **self:** The time spent in the function when it was called specifically by this parent.
- **children:** The time spent in the function's children when it was called by this parent.
- **called:** The number of times this specific parent called the function / the function’s total call count.
#### Child Lines (Below the Function)
- **self:** The time spent directly in this child when called by the current function.
- **children:** The time spent in the child's own descendants when triggered by the current function.
- **called:** The number of times the current function called this child / the child’s total call count.
#### Cycle Entries
- **cycle:** Represents a group of functions that call each other recursively; it lists external callers as parents and cycle members as children.

---
## Common Output Flags
Use these flags to customize the `analysis.txt` file.
- **`-a`**: Suppress information for `static` (private) functions.
- **`-b`**: Brief mode; suppresses verbose explanations of columns.
- **`-p`**: Print only the flat profile.
- **`-P[name]`**: Exclude a specific function from the flat profile.
- **`-q`**: Print only the call graph.
- **`-Q[name]`**: Suppress a specific function from the call graph.

---
## Summary Table

|**Command**|**Action**|
|---|---|
|`gcc -pg ...`|Compiles with profiling data generation.|
|`./a.out`|Runs program and creates `gmon.out`.|
|`gprof [bin] [data]`|Analyzes and outputs profiling results.|
|`gprof -b ...`|Removes the lengthy "help" text from the report.|
