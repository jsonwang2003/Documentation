# Assignment 1 : GDB + x86 _(20 pts)_

**Deadline: Tuesday, October 6th, 2026 by 11:59 PM PDT**

This is a group project; you can work in a team of size at most two and submit one project per team. You are not required to work with the same partner on every project. You and your partner should collaborate closely on each part.

You have three late days that you may use to turn in work past the deadline over the entire quarter. A late day is a contiguous 24-hour period. Both you and your partner will be charged for every late day that you use, and you both must have late days to use them. These late days are intended to cover your extension needs for usual circumstances: brief illness, busy with other classes, interviews, travel, extracurricular conflicts, and so on. You do not need to ask permission to use a late day.

The code and other answers you submit must be entirely your team's own work. You may discuss the conceptualization of the project and the meaning of the questions, but you may not look at any part of someone else's solution or collaborate with anyone other than your partner. You may consult published references, provided that you appropriately cite them (e.g., with program comments).

Solutions must be submitted to Gradescope.

---

## Goal

The goal of this assignment is to become familiar with the setup that will be used for future assignments, such as the use of a virtual machine and the included turn-in script, as well as the basics of working with gdb and writing programs in x86 assembly.

## Getting Started

To complete this assignment, you will be provided with a VirtualBox VM pre-populated with the assignment files.

### VM Image

In order to match the environment in which your submission will be graded, all work for this assignment must be done on the VirtualBox VM we provide, named `pa1box`. You can download the VM image [**here**](https://drive.google.com/file/d/1LDLZmgqZPTE4bhTqloIZXHKHz-9egwsd/view). (For M1/M2 Macs, see the [instructions at the bottom of this page](https://cseweb.ucsd.edu/classes/fa26/cse127-001/pa1/pa1.html#m1-m2-machines).)

The VM is configured with two users: `student` (note that the user name is `student` instead of your school user name), with password `hacktheplanet`; and `root`, with password `hackallthethings`. The VM is configured with SSH on port 2222. Please note that SSH is disabled for `root`, so you can only SSH in as the `student` user. You can still log in as `root` using `su` or by logging into the VM directly.

To SSH into the VM:

```
ssh -p 2222 student@127.0.0.1
```

To copy files from your computer to the VM:

```
scp -P 2222 -r /path/to/files/ student@127.0.0.1:/home/student
```

To copy files from the VM to your computer:

```
scp -P 2222 student@127.0.0.1:/path/to/files/ /destination/path
```

## Part 1: Using GDB _(10 pts)_

Alice is a security engineer in a company called Security4All. She felt that their programs are running a bit weird recently. After doing some research, she suspect that the C compiler they are using might have been backdoored. It is a huge issue because a compromised compiler will potentially defeat all security mechanism they implemented in a higher level language. Now Alice heard that you have experience with gdb in some classes and asks for your help to learn the behavior of a program using gdb.

Files for this sub-assignment are located in the `gdb` subdirectory of the `student` user's home directory in the VM image; that is, `/home/student/gdb`. SSH into the VM and `cd` into that directory to begin working on it.

Inside the `gdb` directory, you'll find `fib.c`, a C program demonstrating the Fibonacci sequence; a `Makefile`; and [hw1.txt](https://cseweb.ucsd.edu/classes/fa26/cse127-001/pa1/hw1.txt), in which you'll record your responses to the questions below. The first step is to compile `fib` by running `make` on the command line.

To run the `fib` executable in GDB, run `gdb fib`. I recommend the following workflow in GDB:

1. **Starting.** Set breakpoints that you can later use for analysis:
    - `b foo` — break at function `foo`
    - `b *0x08048489` — break at the instruction at address 0x08048489
    - `r` — run the executable
2. **Analyzing.** Examine memory, registers, etc; disassemble code; show stack frames, backtrace, etc; and more:
    - `disas foo` — disassemble function `foo`
    - `i r` — view registers
    - `where` — view stack frames
    - `x <loc>` — examine memory
    - `x $eip` — examine current instruction pointer
    - `x /10x $esp` — examine 10 words at top of stack
    - `x /10x buf` — examine 10 words in `buf`
    - `x /10i $eip` — examine 10 instructions starting at instruction pointer
    - `x /10i foo` — examine 10 instructions starting at `foo`
3. **Continuing.** Continue analysis:
    - `c` — continue execution until next breakpoint/watchpoint
    - `si` — step to the next instruction
    - `s` — step to the next line of source code

Note that this is only a cursory overview of GDB; much more info is available from online resources.

### Assignment Instructions

Complete the following exercises and **fill out [hw1.txt](https://cseweb.ucsd.edu/classes/fa26/cse127-001/pa1/hw1.txt)** with your answers. **Follow the directions in the template, do not delete the square brackets**

1. What is the value, in hex, of the `ecx` register when the function `f` is called? _(2 pts)_
2. Which register stores the value of the variable `i` in the function main? _(2 pts)_
3. What is the address, in hex, of the function `f`? _(2 pts)_
4. What is the name of the 6th instruction of the function `f`? _(2 pts)_
5. When `f` completes after being called from `main`, to which address in `main` does control return? Write your answer in hex form. _(2 pts)_

### Submission

Submit `hw1.txt` to "Assignment 1 - GDB" on Gradescope. Gradescope will check that it has successfully detected your answers, but it will not give you your grade until the due date.

```txt title="hw1.txt"
Write your answers in between the square brackets.
They will be used to detect your answer.
DON'T modify anything else in this file!

---
QUESTION #1 - 2 pts

What is the value, in hex, of the ecx register when the function f is called for the first time?

Sample Answer: 0xffffffff
(Your answer should be something like this exactly in 8 hex digits with "0x" as prefix)

Your Answer: [0x7fffffea]

---
QUESTION #2 - 2 pts

Which register stores the value of the variable i in the function main (before f is called)?

Sample Answer: eip

Your Answer: [ebp]

> [!failure] Correct Answer: [eax]

---
QUESTION #3 - 2 pts

What is the address, in hex, of the function f?

Sample Answer: 0xffffffff
(Your answer should be something like this exactly in 8 hex digits with "0x" as prefix)

Your Answer: [0x004005a0]

---
QUESTION #4 - 2 pts

What is the name of the 6th instruction of the function f?

Sample Answer: mov

Your Answer: [add]

---
QUESTION #5 - 2 pts

When f completes after being called from main, to which address in main does control return? Write your answer in hex form.

Sample Answer: 0xffffffff
(Your answer should be something like this exactly in 8 hex digits with "0x" as prefix)

Your Answer: [0x0040063c]

```

---
## Part 2: `echo` in x86 _(10 pts)_

Bob is a developer at Security4All. He received an internal warning that the compiler might be compromised. However, not being able to use a compiler blocks the development of their project that has to be released in a week. While the security team is still investigating the compiler, Bob decides to finish up their development using raw assembly. However, it's been five years since Bob took his undergrad class about assembly, so Bob needs you help to write a simple x86 assembly program to refresh the knowledge.

Files for this sub-assignment are located in the `x86` subdirectory of the `student` user's home directory in the VM image; that is, `/home/student/x86`. SSH into the VM and `cd` into that directory to begin working on it.

For this part, you will be implementing a simplified version of the familiar `echo` command, using raw x86 assembly code. The goal of this assignment is to familiarize you with writing programs directly in x86.

Your `echo` command must behave as follows:

- When run with a single command line argument (e.g., `./echo Hello`):
    1. Prints that argument back to the console's standard output (stdout).
    2. Prints a trailing newline (`\n`).
    3. Exits with code 0.
- When run with too few command line arguments (e.g., `./echo`) or too many (e.g., `./echo Hello World`):
    1. Prints _exactly_ the error message `This command expects exactly one argument.` — followed by a trailing newline (`\n`) — to the console's standard _error_ (stderr).
    2. Exits with code 1.

Your code should be written in the file `echo.s` inside the `x86` directory. A heavily-commented example `echo.s` is provided, which simply prints the message `Hello World` (followed by a trailing newline) to stdout. Your job is to modify this program to meet the specification for `echo` above.

A `Makefile` is included, so you can build the `echo` binary by running `make` from the command line.

### Submission

Submit `echo.s` to "Assignment 1: x86 echo" on Gradescope.

```s title="echo.s"
# Names: Justine Le and Jason Wang
# CSE 127, Assignment 1
# Date: 10/6/26
  .section .rodata

err_msg:
  # Constant holding the message string to print.
  .ascii "This command expects exactly one argument.\n"
  errlen = . -err_msg  # length of err_msg  

  .section .text

  .globl _start
_start:

  ## Print message to standard output.
	
  # 1. check if there's 2 arguments
  # 2. if there is an argument missing --> jump to the bottom where it print the err msg & returns 1
  # 3. if all arguments are there --> then just print normally and exit w 0

  cmp $2,(%esp) # checking if there's 2 args from user
  jne error     # jump if there's not 2 args

  movl 8(%esp), %ecx # ecx points to the string
  movl %ecx, %edx    # edx loops through the string

strlen_loop:
  cmpb $0, (%edx)  # check for null terminator
  je strlen_done   # jump if end of string
  incl %edx        # increase string pointer
  jmp strlen_loop  # jump to loop
strlen_done: 
  addb $'\n', (%edx)
  incl %edx 
  subl %ecx, %edx  # length = end - start

  movl  $4, %eax
  # Select the 'sys_write' syscall.
  # ===============================
  # The value in %eax determines which syscall the kernel will execute when
  # interrupt 0x80 is triggered by our code.

  movl  $1, %ebx
  # Set the output to file descriptor #1 (standard output).
  # =======================================================
  # The value in %ebx is the first argument to the syscall. In the case of
  # sys_write, that first argument is the file descriptor (fd) number to which
  # the string should be written.

  int   $0x80
  # Invoke the syscall and print the message string.
  # ===================
  # This triggers interrupt number 0x80, which the Linux kernel intercepts to
  # run the syscall we select in the %eax register, using the %ebx-%edx
  # registers as its arguments.
  
  jmp no_err # jump to print user's string

 error:
  movl $4, %eax  # setting sys_write syscall
  movl $2, %ebx  # setting fd #2 (stderr)
  movl $err_msg, %ecx  # setting err_mess to print
  movl $errlen, %edx   # setting length of err_mess
  int $0x80 # invoke syscall to print mss
  movl  $1, %eax      # Select the 'sys_exit' syscall
  movl  $1, %ebx      # Set the exit code to 1
  int   $0x80         # Invoke the syscall

 no_err:
  ## Exit the program with code 0 (success).
  movl  $1, %eax      # Select the 'sys_exit' syscall
  movl  $0, %ebx      # Set the exit code to 0
  int   $0x80         # Invoke the syscall
```

---
### Helpful Hints

- In a Linux program, stdout is file descriptor number 1 and stderr is number 2.
- Linux programs start with `argc` at the top of the stack, accessible at `0(%esp)` from x86-assembly programs. Below it is `argv`, the array of pointers to _(null-terminated)_ strings passed into the program as arguments. So `argv[0]` can be accessed from x86-assembly programs at `4(%esp)`, `argv[1]` at `8(%esp)`, `argv[2]` at `12(%esp)`, and so on.
- `argv[0]` is the name of the program executed, _not_ the first argument, which is `argv[1]` if supplied (or `8(%esp)` for our purposes). Then `argc` (or `0(%esp)`) will be 1 if the program was not passed any arguments, 2 if it was passed 1 argument, and so on.
- This [table of Linux system calls](https://www.chromium.org/chromium-os/developer-library/reference/linux-constants/syscalls/) may come in handy.
- For people with AMD CPUs, you may find that your VMs don't boot unless you boot with sysvinit instead of systemd. This is normal, and will not affect your assignment.

### M1/M2 machines

VirtualBox doesn't work on M1/M2 machines. To use the vm image on M1/M2 machines, you will have to install [qemu](https://www.qemu.org/) and run the vm with it.
```bash
brew install qemu

qemu-system-x86_64 \
-cpu qemu64 \ 
-m 4096 \
-device e1000, netdev=net0, mac=08:00:27:D3:23:52 \ 
-netdev user, id=net0, hostfwd=tcp::2222-:22 \ 
-drive file=/Path/to/pa1box/pa1box-disk1.vmdk
```

User and ssh settings are the same as using VirtualBox.

Alternatively, on M1/M2 Macs, install [UTM](https://mac.getutm.app/) and download [this VM image](https://drive.google.com/file/d/1BUKDEE8Y81vFfw3jBi42FqMRGkIIL9rJ/view); after extracting it, you should be able to open it directly in UTM and start the VM.

For SSH access, configure TCP port forwarding from port `2222` on your Mac to port `22` in the VM, then use `ssh -p 2222 student@127.0.0.1`; this [VirtualBox port-forwarding example](https://medium.com/codemonday/ssh-to-virtualbox-port-forwarding-from-wsl2-machine-1f667da18430) illustrates the idea, though UTM's settings are different.