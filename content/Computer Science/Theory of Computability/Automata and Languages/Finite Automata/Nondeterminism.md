> [!Abstract] Formal Definition
> A **Nondeterministic Finite Automaton** is a 5-tuple $(Q, \Sigma, \delta, q_{0}, F)$, where
> 1. $Q$ is a finite set of states
> 2. $\Sigma$ is a finite alphabet
> 3. $\delta$ : $Q \times \Sigma_{\epsilon} \to P(Q)$ is the transition function
> 4. $q_{0} \in Q$ is the start state
> 5. $F \subseteq Q$ is the set of accept states
>    
>  Let $N = (Q, \Sigma, \delta, q_{0}, F)$ be an NFA and $w$ a string over the alphabet $\Sigma$. Then we say that $N$ ***accepts*** $w$ if we can write $w$ as $w = y_{1}y_{2}\dots y_{m}$, where each $y_{i}$ is a member of $\Sigma_{\epsilon}$ and a sequence of states $r_{0}, r_{1}, \dots,r_{m}$ exists in $Q$ with three conditions:
>  1. $r_{0} = q_{0}$
>  2. $r_{i+1} \in \delta(r_{i}, y_{i+1})$, for $i = 0, \dots, m-1$
>  3. $r_{m} \in F$

> [!Note] 
> Unlike [[Computer Science/Theory of Computability/Automata and Languages/Finite Automata/index#How it Computes (Deterministic Computation)|Deterministic Computation]] where the next state is determined to be a specific state, **Nondeterministic Machine** has *several* choices that exists for the next state at any point.
> 
> Nondeterminism is a **generalization** of determinism → every deterministic finite automaton is automatically a nondeterministic finite automaton

## DFA vs NFA

| Deterministic Finite Automaton (DFA)                                                                         | Nondeterministic Finite Automation (NFA)                                  |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Every state **always has exactly one** exiting transition arrow **for each symbol in the alphabet $\Sigma$** | States may have **0, 1, or many** exiting arrows for each alphabet symbol |
| Labels on the transition arrows are symbols from the alphabet $\Sigma$                                       | May have arrows labeled with members of the alphabet **or $\epsilon$**    |
## Computation of NFA
1. **Start:** The process begins in the start state $q_{0}$​.
2. **Read:** The machine reads the input string symbols one by one from left to right.
3. **Move:** After reading a symbol, it follows the transition $\delta$ to a new state.
	- The current state has multiple ways to proceed → the machine "splits" into multiple copies of itself and follows *all* the possibilities in parallel 
	- Each takes one of the possible ways to proceed and continues as before
4. **Output:** After the last symbol is read, the machine **accepts** the string if **any of the copies have an accept state in the final stage**; otherwise, the machine **rejects** the input.

## Equivalence of NFAs and DFAs
- DFAs and NFAs recognize the same class of languages
- Say that two machines are ***equivalent*** if they recognize the same language

### Every NFA Has an Equivalent DFA
#### Proof by Construction
Let $N = (Q, \Sigma, \delta, q_{0}, F)$ be the NFA recognizing some language $A$.

Construct a DFA $M = (Q', \Sigma, \delta', q_{0}', F')$ recognizing $A$. 
1. $Q' = \mathcal{P}(Q)$
	Every state of $M$ is a set of states of $N$
	**Recall** that $\mathcal{P}(Q)$ is the set of subsets of $Q$
2. For $R \in Q'$ and $a \in \Sigma$, let $\delta'(R, a) = \{ q \in Q | q \in \delta(r, a) \text{ for some } r \in R \}$
	If $R$ is a state of $M$, it is also a set of states of $N$
	When $M$ reads a symbol $a$ in state $R$, it shows where $a$ takes each state in $R$, it shows where $a$ takes each state in $R$
	Because each state may go to a set of states, we take the union of all these sets.
	$$
	\delta'(R, a) = \bigcup_{r \in R} \delta (r, a)
	$$
	The notation standards for the **union of the sets $\delta(r, a)$ for each possible $r$ in $R$**

3. $q_{0}' = \{ q_{0} \}$
	$M$ starts in the state corresponding to the collection containing just the start state of $N$
4. $F' = \{ R \in Q' | R \text{ contains an accept state of } N \}$
	The machine $M$ accepts if one of the possible states that $N$ could be in at this point is an accept state.

