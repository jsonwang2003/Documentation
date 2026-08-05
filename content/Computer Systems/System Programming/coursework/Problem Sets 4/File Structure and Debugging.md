## Task
You will demonstrate your proficiency with file system operations, git version control, C programming, debugging, and GDB.

> [!IMPORTANT]
> Before starting any tasks, run these commands in your workspace terminal:
> ```bash
> bash .init_git_repos.sh
> bash .pl_startup_script.sh
> ```
> This will initialize the git repositories install necessary tools

### Task 1 → Create Directory Structure
Create the following directory structure within `~/step1`:
```txt
step1/
├── src/
│   └── main.c
├── lib/
│   └── my_libs.c
└── data/
    ├── 2021/
    │   └── grades.csv
    ├── 2022/
    │   └── grades.csv
    └── 2023/
        └── grades.csv
```
The files can be empty - they just need to exist in the structure

> [!HINT]
> use `mkdir -p` for nested directories and `touch` to create empty files
### Task 2 → Git Commit in Project Directory
The `project` directory is a git repository
1. Create a new file called `README` in the project directory
2. Add it to git
3. Make a commit with a message containing the string `"skill demo"`
Click `"Save and Grade"`
### Task 3 → Run Failing Tests
The `buggy` directory contains `list_examples.c` with bugs in the `filter` and `merge` functions
1. Navigate to the `buggy` directory
2. Run the tests: `./test.sh`
3. Save the test output to `~/test-fail-output.txt`
Click `"Save and Grade"`
### Task 4 → Fix Bugs and Verify
1. Fix the bugs in `list_examples.c`
	- There are 2 bugs in the code
2. Rerun the tests: `./test.sh`
3. Save the successful output to `~/test-success-output.txt`
Click `"Save the Grade"`
### Task 5 → Commit Bug Fixes
Make a git commit in the `buggy` direction with your bug fixes. The commit message must contain the string `"fix bugs"`
### Task 6 → GDB Debugging - Find Malloc Size
The `hash_demo` directory contains `hash_password.c`
#### Using GDB
Debug the program:
```bash
$ cd hash_demo
$ gcc -g -o hash_password hash_password.c
$ gdb ./hash_password
```

Find the size passed to `malloc` for the `hashed` array in the `simple_hash` function (around line 15)
Put this value in `~/hashed.txt`
Click `"Save and Grade"`
### Task 7 → GDB Debugging - Find Array Value
Using the same program and GDB:
1. Set a breakpoint after the for loop completes (around line 28)
2. Run the program with: `run Password123`
3. Print the value of `hashed[1]`
4. Put this value in `~/hashed_index.txt`

---
## Grading Notes
When grading, files are copied to `/grade/student`. If you sees an error about `/grade/student/filename.txt`, it means you didn't create the file in `~/` in your workspace

