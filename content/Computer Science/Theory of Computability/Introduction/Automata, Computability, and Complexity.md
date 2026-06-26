> [!Abstract] Central Question
> 
> What are the fundamental capabilities and limitations of computers?

## Complexity Theory
Computer problems come in different difficulties, some *easy* some *hard*

When confront a problem that *appears to be computationally difficult*, we have several options:
1. Understanding which aspect of the problem is at **the root of the difficulty** can alter it so that the problem is easily solvable
2. May be able to settle for **less than a perfect solution** to the problem (approximate the perfect solution might be relatively easy)
3. Some problems are hard **only in the worst case scenario**, but easy most of the time
	- May be satisfied with a procedure that **occasionally is slow** but **usually runs quickly**
4. May consider **alternative types** of computation, such as *randomized computation*, that can speed up certain tasks

## Computability Theory
Certain basic problems *cannot be solved* by computers
- Determine whether a mathematical statement is true of false

Among the consequences of this result was the development of ideas concerning **theoretical models** of computers that eventually helped to construct actual computers

[[#Computability Theory]] and [[#Complexity Theory]] are closely related
- **Complexity Theory**: classification of problems as *easy* or *hard*
- **Computability Theory**: classification of problems is by those that are *solvable* and *unsolvable*

## Automata Theory
Deals with the **definitions** and **properties** of mathematical models of computation
- [[Computer Science/Theory of Computability/Automata and Languages/index|Finite Automaton]] is used in *text processing*, *compilers*, and *hardware design*
- [[Context-Free Grammar]] is used in *programming languages* and *artificial intelligence*
Automata theory allows practice with **formal definitions of computation** as it introduces concepts relevant to other non-theoretical areas of computer science