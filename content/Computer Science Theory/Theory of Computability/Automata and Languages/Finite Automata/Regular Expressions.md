## 1. Summary Description

A **regular expression** is a way to describe a language using a string of symbols and operators. While a Finite Automaton (DFA/NFA) represents a language through its computational process, a regular expression represents the same language through its structural definition.

## 2. Formal Definition
A value R is a regular expression if it is one of the following:
- $a$ for some $a \in \Sigma$ (a single character from the alphabet).
2. $\epsilon$: Empty String   
3. $\emptyset$: Empty Set
4. $(R_{1} \cup R_{2})$, where $R_{1}$​ and $R_{2}$​ are regular expressions (**Union**).
5. $(R_{1} \circ R_{2})$, where $R_{1}$ and $R_{2}$ are regular expressions (**Concatenation**).
6. $(R_{1}^*)$, where $R_{1}$ is a regular expression (**Star operation**).

> [!Note] Common Confusion between $\epsilon$ and $\emptyset$
> $\epsilon$ represents the language containing a single string  $\to$ the empty string
> 
> $\emptyset$ represents the language that doesn't contain any strings

### Shorthand Notations
For convenience, there are some notations expressed as followed:
- **$R^+$:** shorthand expression for $R R^*$, all strings that are $1$ or more concatenations of strings from $R$
- **$R^k$:** shorthand for the concatenation of $k \ R's$ with each other
- **$L(R)$:** the language of $R$, distinguishing a regular expression $R$ and the language it describes 
## 3. Regular Operations

Regular expressions are built using three fundamental operations:
- **Union ($\cup$):** Corresponds to the logical "OR"
	- $0 \cup 1$ describes the language $\{ 0, 1\}$.
- **Concatenation ($\circ$):** Joining strings together. 
	- Often the $\circ$ symbol is omitted 
	- $ab$ instead of $a \circ b$
- **Star ($^*$):** Represents *zero or more repetitions* of the preceding expression. 
	- $0^*$ describes the language $\{ \epsilon, 0, 00, 000, \dots \}$.
## 4. Precedence of Operations
To avoid excessive parentheses, the operations follow a specific order of precedence:
1. **Star (∗)** has the highest precedence.
2. **Concatenation (∘)** is next.
3. **Union (∪)** has the lowest precedence. 
## 5. Identities
If we let $R$ be any regular expression
	$R \cup \emptyset = R$
	Adding the empty language to any other language will not change it

	$R \circ \epsilon = R$
	Joining the empty string to any string will not change it

However, exchanging $\emptyset$ and $\epsilon$ in the preceding identities may cause the equalities to fail
	$R \cup \epsilon \text{ may not equal } R$
	If $R = 0$, then $L(R) = \{ 0 \}$ but $L(R \cup \epsilon) = \{ 0, \epsilon \}$

	$R \circ \emptyset \text{ may not equal } R$
	If $R = 0$, then $L(R) = \{  0 \}$ but $L(R \circ \emptyset) = \emptyset$

---
# Equivalence with Finite Automata

One of the most important results in formal language theory is that **Regular Expressions and Finite Automata are equivalent in power**.
- **Theorem:** A language is regular if and only if some regular expression describes it.
- **Implication:** For any regular expression, you can build an NFA that recognizes the same language, and for any DFA, you can write a regular expression that describes its language.

---
# Comparison of Representations

| Feature         | Finite Automata (DFA/NFA)            | Regular Expression                  |
| --------------- | ------------------------------------ | ----------------------------------- |
| **Perspective** | Computational (How to process)       | Structural (What it looks like)     |
| **Best Use**    | Implementing search in hardware/code | Writing search queries and patterns |
| **Power**       | Recognizes Regular Languages         | Describes Regular Languages         |