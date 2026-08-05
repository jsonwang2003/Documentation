> [!ABSTRACT]
> 
> This section explores the mechanics of cryptographic hash functions, their role in identifying data uniquely (Fingerprinting), and their application in security systems like Git and SSH.
### 1. Real-World Applications
Hashes are everywhere in systems programming. They act as "IDs" for data that might be too large to handle easily.
#### Git Commit Hashes
When you commit in Git, you receive a 40-character hexadecimal string.

```bash
$ git commit -m "almost done"
[main 211935b] almost done
$ git log
commit: 211935b0ef003...(40 hex char.)
```

- **Computation**: The hash is computed from the **file content** + **commit message**.
- **Integrity**: If a single bit in a file changes, the entire hash changes, ensuring the history cannot be secretly altered.
#### SSH Fingerprints
When connecting to a server, you verify its identity via a fingerprint.

```shell
$ ssh email@ieng6.ucsd.edu
The authenticity of host ieng6 cannot be established
ED25519 key fingerprint is SHA256: 8avDd+0...
```

- **Computation**: This is a hash computed from the **server's public key** on `ieng6`.

---
### 2. Properties of Hash Functions
A hash function takes input of **any size** and returns a **fixed-size integer**. To be considered **cryptographically secure**, it must follow four rules:

| **Property**            | **Description**                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Deterministic**       | The same input **always** produces the exact same output.                                                                                       |
| **One-Way**             | It is computationally infeasible to reverse the process:<br>- Input $\rightarrow$ Hash is easy; <br>- Hash $\rightarrow$ Input is "impossible". |
| **Unpredictable**       | Similar inputs produce **wildly different** hashes (the "Avalanche Effect").                                                                    |
| **Collision-Resistant** | It is extremely improbable for two different inputs to produce the same hash.                                                                   |

> [!NOTE]
> 
> Mathematical vs. Computational Reality
> 
> While it is mathematically impossible to have a truly one-to-one function when mapping infinite inputs to a finite output, $2^{256}$ (~$10^{77}$) is so large that a collision is statistically impossible in the lifetime of the universe.

---
### 3. Comparison of Algorithms

|**Algorithm**|**Size**|**Status**|**Use Case**|
|---|---|---|---|
|**SHA256**|32 Bytes|**Secure**|Modern security, Git (new), Blockchain.|
|**MD5**|16 Bytes|**Insecure**|Checksums for non-secure data; prone to collisions.|

---
### 4. Key Use Cases in Systems
1. **Data Identification**: Identifying large files/commits with a short "ID" string.
2. **Password Management**: Servers should **never** store passwords as plain text. Instead, they store the **hash** of the password. When you log in, the server hashes your attempt and compares it to the stored hash.

---
### Module Navigation
This section links closely with upcoming topics on how the OS manages data and processes.
- **[[Process of Operating System]]**: How the OS uses these concepts for process identity.
- **[[Pointers and Reference]]**: How we handle the memory addresses where these hashes are stored.