---
title: Finite Automata
---
> [!Abstract] Definition
> A **finite automaton** is a $5$-tuple $M = (Q, \Sigma, \delta, q_{0}, F)$, where:
> 1. $Q$ is a finite set called the ***states***
> 2. $\Sigma$ is a finite set called the ***alphabet***
> 3. $\delta: Q \times \Sigma \to Q$ is the ***transition function***
> 4. $q_{0} \in Q$ is the ***start state***
> 5. $F \subseteq Q$ is the ***set of accept states***

## Informal Description
A **finite automaton** (also called a finite state machine) is an idealized model of a computer with a fixed, limited amount of memory.
### Components
- **States:** The machine is always in one of a finite number of states.
	- **Start State** is indicated by **arrow pointing at it from nowhere**
	- **Accept State** is indicated by **double circle**
	- **Transition:** The machine moves from one state to another based on an input symbol it receives.
- **Output:**
	- Either *accept* or *reject*
### Processing
1. Begins in $M_{1}$'s start state
2. Automaton receives *symbols* from the input string **one-by-one** from left to right
3. After reading each symbol $a$, $M_{1}$ moves from one state to another along the **transition** that has the symbol $a$ as its label
4. When the automaton reads **last symbol**, $M_{1}$ produces its output
	- *accept* if $M_{1}$ is now in an *accept* state
	- *reject* if $M_{1}$ is **not** in an *accept* state

> [!Example]
> ![[Pasted image 20260411215121.png]]
> 
> **Input**: $1101$
> 1. Start in state $q_{1}$. 
> 2. Read $1$, follow transition from $q_{1}$ to $q_{2}$. 
> 3. Read $1$, follow transition from $q_{2}$ to $q_{2}$. 
> 4. Read $0$, follow transition from $q_{2}$ to $q_{3}$. 
> 5. Read $1$, follow transition from $q_{3}$ to $q_{2}$. 
> 6. Accept because $M_{1}$ is in an **accept state** $q_{2}$ at the end of the input.

### How it Computes (Deterministic Computation)
1. **Start:** The process begins in the start state $q_{0}$​.
2. **Read:** The machine reads the input string symbols one by one from left to right.
3. **Move:** After reading a symbol, it follows the transition $\delta$ to a new state.
4. **Output:** After the last symbol is read, the machine **accepts** the string if it is in an accept state; otherwise, it **rejects** it.
### Key Terminology
- **State Diagrams:** A visual representation of the automaton where circles represent states and arrows represent transitions.
- **Transition Table:** A tabular representation of the transition function $\delta$, showing the next state for every combination of current state and input symbol.
- **Language of a Machine ($L(M)$):** The set of all strings that the machine $M$ accepts.
- **Recognize:** A machine **recognizes** a language if it accepts every string in that language and rejects all others. Note that while a machine may accept many strings, it recognizes exactly _one_ language.
### Regular Operations
The power of finite automata is often discussed in the context of three "regular operations" used to manipulate languages.
- **Union:** takes all the strings in both $A$ and $B$, lumps them into one language
$$
A \cup B = \{x|x \in A \text{ or } x \in B\}
$$
- **Concatenation:** Attaching strings from one language to strings of another.
$$
A \circ B = \{ xy|x \in A \text{ and } y \in B \}
$$
- **Star:** Repeating strings from a language any number of times.
	- "any number" includes $0$, so the empty string $\epsilon$ is **always** a member of $A^{*}$ 
$$
A^{*} = \{ x_{1}x_{2}\dots x_{k} | k \geq 0 \text{ and each} x_{i} \in A \}
$$

---
## Theorem
### The Class of Regular Languages is Closed Under the Union Operation ($\cup$)

#### Proof by Construction
Assume: 
	$M_{1}$ recognize $A_{1}$ where $M_{1} = (Q_{1}, \Sigma, \delta_{1}, q_{1}, F_{1})$
	$M_{2}$ recognize $A_{2}$ where $M_{2} = (Q_{2}, \Sigma, \delta_{2}, q_{2}, F_{2})$

Construct $M$ to recognize $A_{1} \cup A_{2}$, where $M = (Q, \Sigma, \delta, q_{0}, F)$
1. $Q=\{ (r_{1}, r_{2}) \ | \ r_{1} \in Q_{1} \text{ and } r_{2} \in Q_{2} \}$
	This set is the **Cartesian Product** of sets $Q_{1}$ and $Q_{2}$ and is written $Q_{1} \times Q_{2}$
	It is the set of all pairs of states, the first from $Q_{1}$ and the second from $Q_{2}$
2. $\Sigma$ is the same as in $M_{1}$ and $M_{2}$
3. $\delta$, the transition function, is defined as follows:
	For each $(r_{1}, r_{2}) \in Q$ and each $a \in \Sigma$, let
$$
\delta((r_{1}, r_{2}), a) = (\delta_{1}(r_{1}, a), \delta_{2}(r_{2}, a))
$$
	Hence $\delta$ gets a state of $M$ (which actually is a pair of states from $M_{1}$ and $M_{2}$), together with an input symbol and returns $M$'s next state
4. $q_{0}$ is the pair $(q_{1}, q_{2})$
5. $F$ is the set of pairs in which either member is an accept state of $M_{1}$ or $M_{2}$, written as
$$
F = \{ (r_{1}, r_{2}) \ | \ r_{1} \in F_{1} \text{ or } r_{2} \in F_{2}\}
$$
	This expression is the same as $F=(F_{1} \times Q_{2}) \cup (Q_{1} \times F_{2})$
	Note that is it *not* the same as $F = F_{1} \times F_{2}$