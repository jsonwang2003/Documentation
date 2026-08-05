[Cryptographic hash functions](https://en.wikipedia.org/wiki/Cryptographic_hash_function) take an input of arbitrary length and produces a fixed length output. The special features are that the outputs are _deterministic_ (the same input always generates the same output) and yet the outputs are apparently “random” – two similar inputs are likely to produce very different outputs, and it's difficult to determine the input by looking at the output.

A common application of hashing is in storing passwords. Many systems store only the hash of a user's password along with their username. This ensures that even if someone gains access to the stored data, users' actual passwords are not exposed. When a user types in a password on such a system, the password handling software grants access if the hash value generated from the user's entry matches the hash stored in the password database.

We said above that it is _difficult_ to determine an input given an output. [Password cracking](https://en.wikipedia.org/wiki/Password_cracking) is a family of techniques for accomplishing this difficult (but possible!) task. That is, let's say we have access to a user's password _hash_ only. Can we figure out their password? We could then use it to log in, and it may also be shared across their accounts which we could also access.

In some cases, password cracking can [exploit the structure](https://en.wikipedia.org/wiki/MD5#Security) of a hash function; this is a topic for a cryptography class. In our case, we will take a more brute-force approach: trying variations on existing known passwords, under the assumption that [many passwords are easy to guess](https://en.wikipedia.org/wiki/List_of_the_most_common_passwords).

## Getting Started
To get started, visit the [Github Classroom](https://classroom.github.com/a/DK5n-VdM) assignment link. Select your username from the list (or if you don't see it, you can skip and use your Github username). A repository will be created for you to use for your work. This PA should be completed on the `ieng6` server. Refer [this](https://ucsd-cse29.github.io/fa25/week1/lab1.html#logging-into-ieng6) section in Week 1's Lab for instructions on logging in to your account on `ieng6` and working with files on the server.

> [!NOTE]
> The auto-grader uses the C11 standard of the C programming language

## Overall Task
We'll start by describing the overall task and goal so it's clear where you're going. However, don't start by trying to implement this exactly – use the milestones below to get there.
### `pwcrack` Password Cracker
Write a program `pwcrack` that takes one command-line argument, the `SHA256` hash of a password in hex format.

The program then reads from `stdin` potential passwords, one per line (assumed to be in UTF-8 format).

The program should, for each password:

- Check if the SHA256 hash of the potential password matches the given hash
- Check if the SHA256 hash of the potential password with each of its ASCII characters uppercased or lowercased matches the given hash

If a matching password is found, the program should print

```bash
Found password: SHA256(<matching password>) = <hash>
```

If a matching password is _not_ found, the program should print:

```bash
Did not find a matching password
```

### Compiling Your Code
#### On `ieng6`
Because the OpenSSL libraries on ieng6 are not in the default library search path, you need to use this command:

```bash
gcc -L/usr/lib/x86_64-linux-gnu/ *.c -o pwcrack -lcrypto -lssl
```

The `-L` flag tells the compiler where to find the crypto libraries during linking.
#### On Other Systems
On most other Linux systems or macOS, you can compile with:

```bash
gcc *.c -o pwcrack -lcrypto
```
### Examples
`seCret` has a SHA256 hash of `a2c3b02cb22af83d6d1ead1d4e18d916599be7c2ef2f017169327df1f7c844fd`, and `notinlist` has a SHA256 hash of `21d3162352fdd45e05d052398c1ddb2ca5e9fc0a13e534f3c183f9f5b2f4a041`

```bash
$ ./pwcrack a2c3b02cb22af83d6d1ead1d4e18d916599be7c2ef2f017169327df1f7c844fd
Password
NeverGuessMe!!
secret
Found password: SHA256(seCret) = a2c3b02cb22af83d6d1ead1d4e18d916599be7c2ef2f017169327df1f7c844fd
$ ./pwcrack 21d3162352fdd45e05d052398c1ddb2ca5e9fc0a13e534f3c183f9f5b2f4a041
Password
NeverGuessMe!!
secret
<Press Ctrl-D for end of input>
Did not find a matching password
```

We could also put the potential passwords in a file:

```bash
$ cat guesses.txt
Password
NeverGuessMe!!
secret
$ ./pwcrack a2c3b02cb22af83d6d1ead1d4e18d916599be7c2ef2f017169327df1f7c844fd < guesses.txt
Found password: SHA256(seCret) = a2c3b02cb22af83d6d1ead1d4e18d916599be7c2ef2f017169327df1f7c844fd
```

> [!NOTE]
> we only consider single-character changes when trying uppercase/lowercase variants of the guesses (i.e. we DON'T try all possible capitalizations of the string): In the example below, the correct password is NOT found because going from `SECRET` to `seCret` would require 5 characters to be changed.
> ```bash
> $ ./pwcrack a2c3b02cb22af83d6d1ead1d4e18d916599be7c2ef2f017169327df1f7c844fd
> SECRET
> <Press Ctrl-D for end of input>
> Did not find a matching password
> ```

## Using Problem Set
Most of the needed functionality for the PA is in the problem set problems! So part of your workflow for this PA should be to move code over from the appropriate part of your problem set and into your PA code.
- [[Capitalization Variants]] helps create all the variants of the strings needed for checking
- [[SHA256 Standard Input]] helps with the main loop that does SHA on values from input and has examples of calling SHA256 ([[SHA256 Loop]] and [[SHA256 Argument]] are also related)
- [[SHA256 Hash Checker]] helps with understanding how to compare SHA values
## Design Questions
1. Real password crackers try many more variations than just uppercasing and lowercasing. Do a little research on password cracking and suggest at least 2 other ways to vary a password to crack it. Describe them both, and for each, write a sentence or two about what modifications you would make to your code to implement them. (You don't have to actually implement them).

In my research, I have found something called a "Rainbow Table Attack" where it uses a precomputed table for password cracking. This precomputed table is generated by taking common password and hash them in well known hashing functions (SHA) and store them. To incorporate this, I would include a file with a table (space separated) that has the common password followed by the hashed code. Then I can check if the hashed password is a match with hashed code stored in the table, hence the password crack.

Another technique I found was "Spidering" where the hackers who the individual's information and compile a list of identifying information and common keywords. By using these terms, combined with dictionary attacks, can form high quality password cracking methods. To do this, one would need the information of the individual and place them in a file. Check each keyword and do some alterations to each keyword and see if the hashed password matches

2. How much working memory is needed to store all of the variables needed to execute the password cracker? Based on your response would you say that a password cracker is more memory-limited or is it more limited by how fast the process can run the code?

Based on the implementation of the password cracker, it needs a string to store the input string, the hashed representation of a variation of the input string, and a strign to store the hexadecimal representation of the hahashed password. That in total comes to around 200 in variable memory. Note that there is no need to store all variations of the current password nor the previous password attempt as those can be disregarded as it has failed to match the correct password.

With that, the limiting factor of the password cracker process is the time it takes, or rather, how fast the process can run the code. This is due to the astronomical number of possible combinations for a password is too vast and too random if it is randomly generated. Where the local memory of the program contributes little to the limiting factor.

---
## Code
Repository found [here](https://github.com/ucsd-cse29-fa25/pa2-hashing-and-passwords-jsonwang2003)

```c
#include <stdio.h>
#include <openssl/sha.h>
#include <string.h>

int check_sha256(char* input_string, char* expected_hash){
	unsigned char hash[32];
	SHA256((unsigned char*)input_string, strlen(input_string), hash);

	char result[65];
	for(int i = 0, j = 0; j < 32; i += 2, j++){
		unsigned char high = (hash[j] >> 4) & 0x0f;
		unsigned char low = hash[j] & 0x0f;

		result[i] = high < 10 ? '0' + high : 'a' + (high - 10);
		result[i + 1] = low < 10 ? '0' + low : 'a' + (low - 10);
	}
	result[64] = 0;

	for(int i = 0; i < 64; i++){
		if(result[i] != expected_hash[i]) {return 0;}
	}
	return 1;
}

int main(int argc, char** argv){
	char buffer[100];
	while(1){
		char* maybe_eof = fgets(buffer, sizeof(buffer), stdin);
		if(maybe_eof == NULL){ break; }

		size_t len = strlen(buffer);
		if(len > 0 && buffer[len - 1] == '\n'){ buffer[len - 1] = 0; }
		
		if(check_sha256(buffer, argv[1])){
			printf("Found password: SHA256(%s) = %s\n", buffer, argv[1]);
			return 0;
		}

		for(int i = 0; i < strlen(buffer); i++){
			if(buffer[i] >= 'A' && buffer[i] <= 'Z'){
				buffer[i] += 'a' - 'A';
				if (check_sha256(buffer, argv[1])){ 
					printf("Found password: SHA256(%s) = %s\n", buffer, argv[1]);
					return 0;
				}
				buffer[i] -= 'a' - 'A';
			} else if (buffer[i] >= 'a' && buffer[i] <= 'z'){
				buffer[i] -= 'a' - 'A';
				if(check_sha256(buffer, argv[1])){
					printf("Found password: SHA256(%s) = %s\n", buffer, argv[1]);
					return 0;
				}
				buffer[i] += 'a' - 'A';
			}
		}
	}
	printf("Did not find a matching password\n");
	return 0;
}
```