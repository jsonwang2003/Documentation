Most computer science is about **providing functionality**:
- UI/UX
- Software Architecture
- Operating Systems
- etc

> [!important]
> Computer Security is ***not*** about functionality; its cross cutting

It is about how some implementation of functionality behaves *in the presence of an adversary*

> "Software security is about integrating security practices into the way you build software, not integrating security features into your code" ― Gary McGraw

## Thinking like an Attacker
1. Look for weak links
2. Identify the **assumptions** that security depends on
	- Are they guaranteed to always be true?
	- Can you make them false?
3. Think outside the box

> [!note] This is something that can start to do now and all the time
> When you interact with a system (computerized or not) think about what that system depends on and how it might be exploited

---
# 2 Competing Philosophies about Securing Systems
## Binary Model (Secure vs. Insecure)
- Traditional crypto and trustworthy systems
- Assume adversary limitations $X$ and define security policy $Y$
- If $Y$ cannot be violated without needing $X$ then system is secure, else insecure
- You know people are invoking some version of this model if they say "proof of security", "secure by design", or "trustworthy systems"

> [!example] Perfect Substitution Cipher
> Choose a string of **random** bits the same length as the plaintext (key), XOR them with plaintext to obtain the ciphertext (Assume key is only known to sender and receiver)
> 
> Perfect Secrecy (proved by Claude Shannon)
> - Probability that a given message is encoded in ciphertext is unaltered by knowledge of the ciphertext
> - Proof: Given any plaintext message and any ciphertext and I can construct a key that will produce the ciphertext from the plaintext → There will be **zero information** in ciphertext
### Problems with the Binary Model
#### Abstract Design != Concrete Artifact
Many assumptions are **brittle** in real systems
- Real artifacts fragile, imperfect, have bugs / limitations
  Don't do precisely what spec or documentation says
- Large gap between **abstraction** and **implementation**
#### Security Evolution
Engineers often delude into thinking that we understand our own creations, or that we can create complex systems to do only what we meant them to do. However, nobody knows how these systems really work. Complexity of computer systems is approaching complexity of biological organisms
- 3 billion base pairs in human genome
- 10+ billion transistors in modern CPUs
Complex systems co-evolve with attacks against them
- How we use systems, how we depend on them and how they might be attacked ― all change over time
- Systems deemed secure today may not be resilient to new threats
## Risk Management Model (More Secure vs. Less Secure)
- Most commercial software development (and much real-world security)
- Try to identify most significant risks to security and minimize them
- Improve security where cost effective (expected value)
- You know people are in this model if they use the words "risk", "mitigation", "defenses", "resilience", etc

> [!example] Concrete Barricades
> The large pillars on the side of the road was made to prevent vehicles from storming into the sidewalks. There are cost effective compared to creating a whole concrete wall while serving the same purpose

### Problems with the Risk Management Model
#### You Never Win
Can just produce stalemate
![[Pasted image 20260924151604.png]]

#### How to Measure?
Its finite to say security is a spectrum, but how to **evaluate** risk or reward?
- How many units of security does your anti-virus or EDR product give you?

> [!question] How do we measure security
> Unlike real world security scenarios such as drug and car safety where they can study user / test experience to give a quantitative measure of how secure something is, computer security risks tend not to be public and has no agency to manage the overall environment of computer security. As such, computer security has no defined measurable unit.

