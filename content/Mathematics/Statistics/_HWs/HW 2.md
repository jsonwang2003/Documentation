## Question 1: Drawing slips from a hat

A hat contains slips of paper numbered 1 through 6. You draw two slips of paper at random from the hat, without replacing the first slip into the hat.

### 1(a) (5 points) Write out the sample space $S$ for this experiment
The sample space $S$ is the set of all possible results of the 2 slips of paper's value. Since the first slip will not be replaced, it is impossible to get 2 of the same value

$$
\begin{align*}
S = \{\\
&(1,2),(1,3),(1,4),(1,5),(1,6), \\
&(2,1),(2,3),(2,4),(2,5),(2,6), \\
&(3,1),(3,2),(3,4),(3,5),(3,6), \\
&(4,1),(4,2),(4,3),(4,5),(4,6), \\
&(5,1),(5,2),(5,3),(5,4),(5,6), \\
&(6,1),(6,2),(6,3),(6,4),(6,5) \\
\}
\end{align*}
$$

### 1(b) (5 points) Express the event $E : \{\text{the sum of the numbers on the slips of paper is } 4\}$ as a subset of $S$

$$
E = \{(1, 3), (3, 1)\}
$$
### 1(c) (5 points) Find $P(E)$

$$
P(E) = \frac{|E|}{|S|} = \frac{2}{30} = \frac{1}{15} \approx \boxed{0.067}
$$
### 1(d) (5 points) Let $F = \{\text{the larger minus the smaller number is } 0\}$. What is $P(F)$

To satisfy this event, the larger number must be the same as the smaller number for their difference to be 0. Since there is no duplicates in $S$, the event space $F$ is:

$$
F = \{\emptyset\}
$$

As such the probability for $F$ is:

$$
P(F) = \frac{|F|}{|S|} = \frac{0}{30} = \boxed{0}
$$

### 1(e) (5 points) Are $E$ and $F$ disjoint? Why or why not

To find disjoint, we try to find the intersection between $E$ and $F$

$$
E \cap F = {\emptyset}
$$
Since there is no common results between $E$ and $F$, they are disjoint
### 1(f) (5 points) Find $P(E \cup F)$
Using the an axiom of probability, if $E$ and $F$ are disjoint, then the probability of $E$ or $F$ is:

$$
P(E \cap F) = P(E) + P(F) = \frac{2}{30} + 0 \approx \boxed{0.067}
$$

---
## Question 2: Cookie selection

### (10 points) A box of cookies contains 5 chocolate chip and 10 sugar cookies. If 5 cookies are randomly selected, what is the probability that three are chocolate chip and two are sugar cookies

First we find the number of outcomes that satisfies the given condition: 

$$
E = \{\text{3 chocolate chip cookie and 2 sugar}\} = \binom{10}{2}\binom{5}{3}
$$
where:
- $\binom{10}{2}$ represents the number of outcomes to choose 2 sugar cookies at random from 10 total sugar cookies
- $5 \choose 3$ represents the number of outcomes to choose 3 chocolate chip cookies from 5 total chocolate chip cookies

Next, we find the total sample space to choose 5 cookies out of 15 total cookies:

$$
S = \binom{15}{5}
$$
As such the probability for the event $E$ will be:

$$
P(E) = \frac{|E|}{|S|} = \frac{\binom{10}{2}\binom{5}{3}}{\binom{15}{5}} = \frac{450}{3003} \approx \boxed{0.15}
$$

---
## Question 3: Inclusion–Exclusion principle for three events

Given three events $A, B, C \subset S$, we want the probability of

$$
P(A \cup B \cup C)
$$

Let $E = A \cup B \cup C$.

### 3(a) (5 points) Define the event $D = A \cup B$. What is the event $C \cup D$ in relation to $E$

The set $E$ was defined to be: "$A$ or $B$ or $C$ occurred"
The expression $D$, on the other hand, represents "A or B occurred"

So:

$$
C \cup D = C \cup (A \cup B) = A \cup B \cup C = E
$$
Therefore the expression $C \cup D$ is equivalent to the event $E$

### 3(b) (5 points) Using Rule 5, write down the expression for $P(C \cup D)$

$$
P(C \cup D) = P(C) + P(D) - P(C \cap D)
$$

### 3(c) (5 points) In words, what does the event $C \cap D$ represent

This expression represents "both C and at least A or B occurred"

### 3(d) (5 points) Show that $C \cap D = (A \cap C) \cup (B \cap C)$ using set identities or a Venn diagram

![[Pasted image 20260122134425.png]]

With this image, we can see the Venn Diagram of the variables $A$, $B$ and $C$:
1. $D$ covers the areas $\{A, B, A \cap C, A \cap B, B \cap C, A \cap B \cap C\}$
2. As such,  $C \cap D$ covers the areas in $\{A \cap C, B \cap C, A \cap B \cap C\}$

When viewing it again in a different perspective
1. $(A \cap C) \cup (B \cap C)$ covers all intersections that was mentioned in $C \cap D$

With that, it is clear that the statement $C \cap D = (A \cap C) \cup (B \cap C)$ is true

### 3(e) (5 points) Using the identity above, use Rule 5 to write down the expression for $P(C \cap D)$

$$
\begin{align*}
P(C \cap D) &= P((A \cap C) \cup (B \cap C))\\
&= P(A \cap C) + P(B \cap C) - P((A \cap C) \cap (B \cap C))

\end{align*}
$$
### 3(f) (5 points) In words, what does $(A \cap C) \cap (B \cap C)$ represent? Is it the same as $A \cap B \cap C$

This statement represents "both $A$ and $C$ occurred and both $B$ and $C$ also occurred". Both $A \cap C$ and $B \cap C$ guarantees the occurrence of $C$, therefore this statement is logically the same as $A \cap B \cap C$

### 3(g) (5 points) Combine your answers from (b) and (e) to write down the final expression for $P(E)$

$$
\begin{align*}
P(E) &= P(C) + P(D) - P(C \cap D)\\
&= P(C) + P(A \cup B) - P(A \cap C) - P(B \cap C) + P((A \cap C) \cap (B \cap C))\\
&= \boxed{P(C) + P(A) + P(B) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)}
\end{align*}
$$
---

## Question 4: Conditional probability with a union

### (10 points) Let $A, B \subset S$ be two events. Show that $P(A \cup B \mid B) = 1$

You may argue algebraically from the axioms or with a Venn diagram.

![[Pasted image 20260122140250.png]]

From the image, the set $A \cup B$ represents the space that is either green or red or both. With the expression $P(A \cup B | B)$ suggests that we know $B$ has already occurred. But because $B$ is a subset of $A \cup B$, the event $A \cup B$ has already occurred by the given. Therefore the probability $P(A \cup B | B) = 1$ is accurate 

---

## Question 5: Two dice (one white, one red)

Events:
$$
\begin{align*}
&A = \{\text{The sum is } 7\}, \\
&B = \{\text{The white die is odd}\},\\
&C = \{\text{The number on the red die is greater than the number on the white die}\}\\
&D = \{\text{The number on both dice are the same}\}
\end{align*}
$$

### 5(a) (5 points) Which pair(s) of events among $A,B,C,D$ are disjoint

The events $A$ and $D$ are disjoint

### 5(b) (5 points) Which pair(s) of events among $A,B,C,D$ are independent

The events $B$ and $D$ are independent

### 5(c) (5 points) In words, what does the event $A \cap B$ represent, and compute $P(A \cap B)$

The event $A \cap B$ represents the 2 dice has a sum of $7$ where the white dice rolled an odd number

The probability of this event is:

$$
P(A \cap B) = \frac{|A \cap B|}{|S|} = \frac{3}{36} = \boxed{\frac{1}{12}}
$$

### 5(d) (5 points) In words, what does the event $A \cap D$ represent, and compute $P(A \cap D)$

The event $A \cap D$ represents the 2 dice has a sum of $7$ and rolled the the same number

The probability of the event is:

$$
P(A \cap D) = \frac{|A \cap D|}{|S|} = \frac{0}{36} = \boxed{0}
$$

### 5(e) (5 points) In words, what does the event $D \mid B$ represent, and compute $P(D \mid B)$

The event $D | B$ represents the 2 dice rolled the same number given that the white dice rolled an odd number

The probability of the event is:

$$
P(D | B) = \frac{|D \cap B|}{|B|} = \frac{3}{18} = \boxed{\frac{1}{6}}
$$

### 5(f) (5 points) In words, what does the event $B \mid D$ represent, and compute $P(B \mid D)$ (Hint: Use Bayes’ Rule)

The event $B|D$ represents the white dice rolled an odd number given that both dice rolled the same value.

By using the Bayes theorem, we can find the probability of this event to be:

$$
P(B|D) = \frac{P(D|B)P(B)}{P(D)} = \frac{\frac{1}{6} \cdot \frac{1}{2}}{\frac{1}{6}} = \boxed{\frac{1}{2}}
$$

---

## Question 6: Casino D6 game

At a casino game, you roll a D6 die and, based on the outcome $i$, you win or lose

$$
10 \times i \times (i - 3)
$$

dollars. Let $X$ be the random variable denoting your winning/loss.

### 6(a) (5 points) What is $\text{supp}(X)$

$$
supp(X) = \{-20, 0, 40, 100, 180\}
$$

### 6(b) (5 points) Write down the probability mass function $P_X(x)$

$$
P_X(x) = \begin{cases}
\frac{2}{6}, &x = -20\\
\frac{1}{6}, &x = 0\\
\frac{1}{6}, &x = 40\\
\frac{1}{6}, &x = 100\\
\frac{1}{6}, &x = 180
\end{cases}
$$

### 6(c) (5 points) Express the event “you don’t win any money” in terms of $X$

$$
A = \{\text{You don't win any money}\} = (X \leq 0)
$$

### 6(d) (5 points) What is the probability of this event

$$
P(A) = P(X \leq 0) = P(X = -20) + P(X = 0) = \frac{2}{6} + \frac{1}{6} = \frac{3}{6} = \boxed{\frac{1}{2}}
$$

### 6(e) (5 points) Calculate your expected winning $E(X)$

$$
\begin{align*}
E(X) = &\sum_{x \in supp(X)} x \cdot P(X = x)\\
= &(-20 \cdot P(X = -20)) + (0 \cdot P(X = 0)) + (40 \cdot P(X = 40)) + \\
&(100 \cdot P(X = 100)) + (180 \cdot P(X = 180))\\
= &(-20 \cdot \frac{2}{6}) + (0) + (40 \cdot \frac{1}{6}) + (100 \cdot \frac{1}{6}) + (180 \cdot \frac{1}{6})\\
= &\frac{-40}{6} + \frac{40}{6} + \frac{100}{6} + \frac{180}{6}\\
= &\frac{-40 + 40 + 100 + 180}{6}\\
= &\frac{280}{6}\\
\approx &\boxed{46.67}
\end{align*}
$$

### 6(f) (10 points) What is the variance $\text{Var}(X)$ and what unit of measurement is $\text{Var}(X)$ in

$$
\begin{align*}
Var(X) = &\sum_{x \in supp(X)} P(X = x) \cdot (x - E(X))^2\\
= &(\frac{2}{6} \cdot (-40 - \frac{280}{6})^2) + (\frac{1}{6} \cdot (0 - \frac{280}{6})^2) \\
&+ (\frac{1}{6} \cdot (40 - \frac{280}{6})^2 + (\frac{1}{6} \cdot (100 - \frac{280}{6})^2)) + (\frac{1}{6} \cdot (180 - \frac{280}{6})^2)\\
= &2503.7 + 362.96 + 7.41 + 474.07 + 2962.96\\
= &\boxed{6311.1}
\end{align*}
$$

The unit of $Var(X)$ is **dollar squared**

---
## Question 7: Variance of a linear combination

In class we saw that the expectation is linear: for random variables $X, Y$ and $a, b \in \mathbb{R}$,
$$
E(aX + bY) = aE(X) + bE(Y)
$$

Assume $X \perp\!\!\!\perp Y$.

### 7(a) (5 points) Expand the square $(aX + bY)^2$ as a quadratic expression in $X$ and $Y$

$$
(aX + bY)^2 = a^2X^2 + b^2Y^2 + 2abXY
$$

### 7(b) (5 points) Using Equation (1) and expansion, what is $(E(aX + bY))^2$

$$
(E(aX + bY))^2 = (aE(X) + bE(Y))^2 = \boxed{a^2E(X)^2 + b^2E(Y)^2 + 2abE(X)E(Y)}
$$

### 7(c) (5 points) Write the alternate expression for $\text{Var}(aX + bY)$ and substitute your answers from (a) and (b) to obtain a final expression

$$
\begin{align*}
Var(aX + bY) = &E((aX + bY)^2) - E(aX + bY)^2\\
= &E(a^2X^2 + b^2Y^2 + 2abXY) - (a^2E(X)^2 + b^2E(Y)^2 + 2abE(X)E(Y))\\
= &a^2E(X^2) + b^2E(Y^2) + 2abE(XY) \\
&- (a^2E(X)^2 + b^2E(Y)^2 + 2abE(X)E(Y))\\
= &\boxed{a^2(E(X^2) - E(X)^2) + b^2(E(Y^2) - E(Y)^2) + 2ab(E(XY) - E(X)E(Y))}
\end{align*}
$$

### 7(d) (10 points) Using independence and $E(XY) = E(X)E(Y)$, show that $\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y)$

$$
\begin{align*}
Var(aX + bY) &= a^2(E(X^2) - E(X)^2) + b^2(E(Y^2) - E(Y)^2) + 2ab(E(XY) - E(X)E(Y))\\
&= a^2(E(X^2) - E(X)^2) + b^2(E(Y^2) - E(Y)^2) + 2ab(E(XY) - E(XY))\\
&= a^2(E(X^2) - E(X)^2) + b^2(E(Y^2) - E(Y)^2) + 2ab(0)\\
&= a^2(E(X^2) - E(X)^2) + b^2(E(Y^2) - E(Y)^2)\\
&= a^2(Var(X)) + b^2(Var(Y))\\
&= \boxed{a^2Var(X) + b^2Var(Y)}
\end{align*}
$$