## 1

Let $X_1, X_2, \ldots, X_n$ be $iid$ random variables with expected value $E(X_i) = \mu$ and variance $\operatorname{Var}(X_i) = \sigma^2$ for every $1 \le i \le n$. Let $\bar{X}$ be given by

$$
\bar{X} = \frac{X_1 + X_2 + \cdots + X_n}{n}.
$$

### (a) (5 points)

Show that we can write $\bar{X} = \sum_{i=1}^n a_i X_i$ for some $a_1, a_2, \ldots, a_n \in \mathbb{R}$. What are the values of $a_1, a_2, \ldots, a_n$?

**SOLUTION**

$$
\begin{align*}
\overline{X} &= \frac{X_1 + X_2 + \cdots + X_n}{n}\\
&= \frac{X_1}{n} + \frac{X_2}{n} + \cdots + \frac{X_n}{n}\\
&= \sum_{i=1}^n \boxed{\frac{1}{n}}(X_i)
\end{align*}
$$

the values for $a_i = \frac{1}{n}$

### (b) (5 points)

Using your answer for part (a) and using the linearity from Equation (1) from HW-2, show:

$$
E(\bar{X}) = \mu.
$$

**SOLUTION**

$$
\begin{align*}
E(\overline{X}) &= E(\sum_{i=1}^n \frac{1}{n}X_i)\\
&= E(\frac{X_1}{n} + \frac{X_2}{n} + \cdots + \frac{X_n}{n})\\
&= \frac{1}{n}E(X_1) + \frac{1}{n}E(X_2) + \cdots + \frac{1}{n}E(X_n)\\
&= \frac{1}{n}\sum_{i = 1}^n E(X_i)\\
&= \frac{1}{n} \sum_{i=1}^n \mu\\
&= \frac{1}{n} (n \cdot \mu)\\
&= \boxed{\mu}
\end{align*}
$$

### (c) (10 points)

Using Equation (2) from HW-2 and your answer from part (a), show that

$$
\operatorname{Var}(\bar{X}) = \frac{\sigma^2}{n}.
$$

**SOLUTION**

$$
\begin{align*}
Var(\overline(X)) &= Var(\sum_{i=1}^n \frac{1}{n}X_i)\\
&= Var(\frac{X_1}{n} + \frac{X_2}{n} + \cdots + \frac{X_n}{n})\\
&= \frac{1}{n^2}Var(X_1) + \frac{1}{n^2}Var(X_2) + \cdots + \frac{1}{n^2}Var(X_n)\\
&= \frac{1}{n^2}\sum_{i = 1}^nVar(X_i)\\
&= \frac{1}{n^2}\sum_{i=1}^n\sigma^2\\
&= \frac{1}{n^2}(n \cdot \sigma^2)\\
&= \boxed{\frac{\sigma^2}{n}}
\end{align*}
$$

---
## 2

Let $X$ be a discrete random variable with probability mass function given by

$$
p_X(x) =
\begin{cases}
C/4 & \text{if } x = 0,\\
C/2 & \text{if } x = 1,\\
C   & \text{if } x = 2,\\
0   & \text{otherwise.}
\end{cases}
$$

### (a) (5 points)

Find the value of $C$ that makes $p_X$ a valid probability mass function.

**SOLUTION**

$$
\begin{align*}
\sum_{x \in supp(X)} p_X(x) &= 1\\
p_X(X = 0) + p_X(X = 1) + p_X(X=2) + p_X(X \neq 0, 1, 2) &= 1\\
\frac{C}{4} + \frac{C}{2} + C + 0 &= 1\\
C + 2C + 4C &= 4\\
7C &= 4\\
C &= \boxed{\frac{4}{7}}
\end{align*}
$$
### (b) (5 points)

Using the value of $C$ you obtained, find the expected value $E(Y)$ where $Y = (X - 1)^2$. What is the support $\operatorname{supp}(Y)$?

**SOLUTION**

$$
\begin{align*}
E(f(x)) &= \sum_{x \in supp(X)} \mathbb{P}(X = x) f(x)\\
E(Y) &= \sum_{x \in supp(X)} \mathbb{P}(X = x)(x - 1)^2\\
&= P(X = 0)(0-1)^2 + P(X = 1)(1 - 1)^2 + P(X = 2)(2 - 1)^2\\
&= \frac{1}{7}(1) + \frac{2}{7}(0) + \frac{4}{7}(1)\\
&= \frac{1}{7} + \frac{4}{7}\\
&= \boxed{\frac{5}{7}}
\end{align*}
$$

---

## 3 (10 points)

A die is tossed until the first six occurs. What is the probability that it takes 4 or more tosses?

**SOLUTION**
Let $N = \text{number of tosses until the first 6 appears}$
- $P(\text{rolling a 6}) = \frac{1}{6}$
- $P(\text{not rolling a 6}) = \frac{5}{6}$

This is a Geometric Distribution where rolling "$4$ or more tosses" means the first $6$ does not appear on tosses $1$, $2$, and $3$

Need to find the CDF: 

$$
F_N(n) = P(N < 4) = (\frac{5}{6})^3 = \frac{125}{216} = \boxed{0.579}
$$

---
## 4

You reach the Nobel Drive station at 3:45 PM to catch the Blue Line trolley to campus. The screen at the terminal says that the trolley is scheduled to arrive at 4:00 PM. With probability 0.1 the trolley is running ahead of schedule and will arrive 5 minutes early. With probability 0.7 the trolley is running late and will arrive 10 minutes late. With the remaining 0.2 probability, the trolley is on time. Let $X$ be the random variable denoting the arrival time of the trolley.

### (a) (5 points)

What is the support of $X$, i.e., $\operatorname{supp}(X)$?

**SOLUTION**

$$
supp(X) = \{-5, 0, 10\}
$$

where:
- $-5$ = arriving $5$ minutes early
- $0$ = arriving on time
- $10$ = arriving late

### (b) (5 points)

Write down the probability mass function $p_X$.

**SOLUTION**

$$
P_X = \begin{cases}
0.1 &\text{if } X = -5\\
0.2 &\text{if } X = 0\\
0.7 &\text{if } X = 10\\
0 &\text{otherwise}
\end{cases}
$$

### (c) (5 points)

Calculate the expected arrival time $E(X)$.

**SOLUTION**

$$
\begin{align*}
E(X) &= \sum_{x \in supp(X)} p_X(x)\cdot x\\
&= (0.1)(-5) + (0.2)(0) + (0.7)(10)\\
&= -0.5 + 0 + 7\\
&= \boxed{6.5}
\end{align*}
$$

## (continued) 
Let $Y$ be the random variable denoting the amount of time you have to wait for the trolley in minutes.

### (a) (5 points)

Write down the support of $Y$, i.e., $\operatorname{supp}(Y)$.

**SOLUTION**

$$
supp(Y) = \{10, 15, 25\}
$$

where :
- $10$ = arriving early
- $15$ = arriving on time
- $25$ = arriving late

### (b) (5 points)

Compute the expected value of $Y$, i.e., $E(Y)$.

**SOLUTION**

$$
\begin{align*}
E(Y) &= \sum_{y \in supp(Y)} P(Y = y) \cdot y\\
&= P(Y = 10)(10) + P(Y = 15) (15) + P(Y = 25)(25)\\
&= (0.1)(10) + (0.2)(15) + (0.7)(25)\\
&= 1 + 3 + 17.5\\
&= \boxed{21.5}
\end{align*}
$$
### (c) (5 points)

Compute $\operatorname{Var}(Y)$.

**SOLUTION**

$$
Var(Y) = E(Y^2) - E(Y)^2
$$

Find $E(Y^2)$

$$
\begin{align*}
E(Y^2) &= \sum_{y \in supp(Y)}P(Y = y) y^2\\
&= P(Y = 10)(10)^2 + P(Y = 15)(15)^2 + P(Y = 25)(25)^2\\
&= (0.1)(100) + (0.2)(225) + (0.7)(625)\\
&= 10 + 45 + 427.5\\
&= \boxed{492.5}
\end{align*}
$$

Use this result and plug it back into the variance formula:

$$
\begin{align*}
Var(Y) &= E(Y^2) - E(Y)^2\\
&= 492.5 - 21.5^2\\
&= 492.5 - 462.25\\
&= \boxed{30.25}
\end{align*}
$$
### (d) (5 points)

What are the units of $\operatorname{Var}(Y)$?

**SOLUTION** The unit of $Var(Y)$ should be minutes squared ($\text{minute}^2$)

---
## 5

Luke Kennard of the Memphis Grizzlies has a 3-point percentage (probability of making a 3-point shot) of $p = 43.9\%$.

### (a) (5 points)

Let $X$ be a random variable that represents whether Luke makes a 3-point shot or not. Identify the most appropriate distribution for $X$. Write down $\operatorname{supp}(X)$ and $p_X(x)$.

**SOLUTION**

$$
supp(X) = \{0, 1\}
$$

$$
p_X(x) = \begin{cases}
0.439 &x = 1\\
0.561 &x = 0
\end{cases}
$$

### (b) (5 points)

Let $Y$ be the number of attempts Luke needs to take in order to successfully make his first 3-point shot in the game. Identify the most appropriate distribution to model $Y$.

**SOLUTION**

Since the interest is how many trials before the first success, the most suitable model is the following:

$$
Y \sim Geo(p = 0.439)
$$

### (c) (5 points)

Find the expected value of $Y$ and its variance.

**SOLUTION**

$$
E(Y) = \frac{1}{p} = \frac{1}{0.429} \approx \boxed{2.28}
$$

$$
Var(Y) = \frac{1-p}{p^2} = \frac{1-0.439}{0.4395^2} \approx \boxed{2.91}
$$
### (d) (5 points)

In the upcoming game against Orlando Magic, suppose Luke attempts $n = 50$ 3-point shots. Let $Z$ be the number of successful 3-point shots Luke makes in these 50 attempts. What is the most appropriate distribution to model $Z$? Write down its PMF $p_Z(z)$.

**SOLUTION**

Since the random variable $Z$ now represents the number of success out of $n = 50$ trials, $Z$ can be expressed by the following distribution:

$$
Z \sim Bin(n = 50, p = 0.439)
$$

The Probability Mass Function is as follows:

$$
P(Z = k) = \binom{n}{k}p^k(1-p)^{n-k} = \binom{50}{k} (0.439)^k(0.561)^{50-k} 
$$

for any $0 \leq k \leq 50$
### (e) (5 points)

Find the expected value of $Z$ and its variance.

**SOLUTION**

$$
E(Z) = n \cdot p = (50)(0.439) = 21.95
$$

$$
Var(Z) = n \cdot p \cdot (1-p) = (50)(0.439)(0.561) = 12.314
$$

### (f) (5 points)

Steph Curry of the Golden State Warriors has a 3-point percentage of $p = 42.6\%$. In the upcoming game against the Clippers, suppose Steph Curry attempts $n = 50$ shots. Let $W$ be the number of successful 3-point shots Steph Curry makes in these 50 attempts. Do you expect $Z$ to be greater than, less than, or equal to $W$? Justify your answer.

**SOLUTION**

Since Steph's probability of success is lower, I expect $Z$ to be greater than $W$ as the expected value grows as the probability of success grows

### (g) (5 points)

What is the expected number of 3-point shots Steph Curry needs to attempt in order to successfully make his first 3-point shot in the game?

**SOLUTION**

Let $V = \text{number of trails before the first success}$

$$
V \sim Geo(p = 0.426)
$$

$$
E(V) = \frac{1}{p} = \frac{1}{0.426} = \boxed{2.35}
$$

### (h) (5 points)

Suppose Steph Curry and Luke Kennard face off to take a single 3-point shot. Let $X_S$ and $X_L$ be the random variables representing the outcome of the shot for Steph Curry and Luke Kennard, respectively. In words, what does the event $A = \{X_S > X_L\}$ represent? Compute $P(A)$.

**SOLUTION**

The event $A = \{\text{Steph Curry made the shot and Luke did not}\}$ because the random variables $X_S$ and $X_L$ represents a single Bernoulli trial with the support of $0$ or $1$. To have the event $X_S > X_L$, the random variables must follow the pattern to satisfy the condition

$$
\begin{align*}
P(A) &= p_S \cdot (1 - p_L)\\
&= (0.426) \cdot (1-0.439)\\
&= \boxed{0.239}
\end{align*}
$$

---
## 6

Let $X$ and $Y$ be two independent $\operatorname{Unif}(0, 1)$ random variables. Let $Z = \max\{X, Y\}$ be a new random variable which is the maximum of $X$ and $Y$. In this question, we’ll derive the PDF $f_Z(z)$ for $Z$.

### (a) (5 points)

Write down the expression for $F_X(x)$, where $F_X(x)$ is the CDF of $X$ for $x \in [0, 1]$.

**SOLUTION**

$$
F_X(x) = \begin{cases}
0 & x < 0\\
x & 0 \leq x \leq 1\\
1 & x > 1
\end{cases}
$$

### (b) (5 points)

What is $\operatorname{supp}(Z)$?

**SOLUTION**

$$
supp(Z) = supp(max(X, Y)) = \{0, 1\}
$$

### (c) (5 points)

For some fixed $z \in \operatorname{supp}(Z)$, consider the events $\{X \le z\}$, $\{Y \le z\}$. In words, describe the event $\{X \le z\} \cap \{Y \le z\}$.

**SOLUTION**

The expression $\{X \leq z\}\cap \{Y \leq z\}$ reflects when both $\{X \leq z\}$ and $\{Y \leq z\}$. As such, the expression represents the event when both random variables $X$ and $Y$ are at most $z$

### (d) (5 points)

Using the independence of $X$ and $Y$, write down the expression for

$$
P(\{X \le z\} \cap \{Y \le z\})
$$

in terms of the CDFs $F_X$ and $F_Y$, respectively.

**SOLUTION**

$$
\begin{align*}
P(\{X \leq z\} \cap \{Y \leq z\}) &= P(X \leq z) P(Y \leq z)\\ &= \boxed{F_X(z) \cdot F_X(z)}
\end{align*}
$$

### (e) (5 points)

Using the expression for the CDF of the uniform distribution from part (a), derive the final expression for

$$
P(\{X \le z\} \cap \{Y \le z\}).
$$

**SOLUTION**

$$
\begin{align*}
P(\{X \leq z\} \cap \{Y \leq z\}) &= F_X(z)F_Y(z)\\
&= z \cdot z\\
&= \boxed{z^2}
\end{align*}
$$

### (f) (5 points)

How is the event $\{X \le z\} \cap \{Y \le z\}$ related to the event $\{Z \le z\}$? Based on your answer, write down the expression for the CDF $F_Z(z)$, i.e.,

$$
F_Z(z) = P(Z \le z).
$$

**SOLUTION**

Since $Z = max(X, Y)$, the mapping for the sample space will be the same between $\{Z \leq z\}$ and $\{X \leq z\}\cap \{Y \leq z\}$ 

$$
F_Z(z) = P(Z \leq z) = P(X \leq z, Y \leq z) = \begin{cases}
0 & z < 0\\
z^2 & 0 \leq z \leq 1\\
1 & z > 1\\
\end{cases}
$$

### (g) (10 points)

Now, using the relationship between the CDF and the PDF from the lecture (fundamental theorem of calculus), derive the expression for the PDF $f_Z(z)$.

**SOLUTION**

$$
\begin{align*}
f_Z(z) = \begin{cases}
\frac{d}{dx}(0) & z < 0\\
\frac{d}{dx}(z^2) & 0 \leq z \leq 1\\
\frac{d}{dx}(1) & z > 1
\end{cases}\\
\boxed{f_Z(z) = \begin{cases}
0 & z < 0\\
2z & 0 \leq z \leq 1\\
0 & z > 1
\end{cases}}
\end{align*}
$$

---
## 7 (5 points)

Give an example of a probability density function $f_X(x)$ whose associated continuous random variable $X$ has expected value 10, i.e., $E(X) = 10$.

**SOLUTION**

Let $X \sim Unif(5, 15)$ 


Given this continuous random distribution, the probability density function can be defined as:

$$
f_X(x) = \begin{cases}
\frac{1}{10} & 5 \leq x \leq 15\\
0 & \text{otherwise}
\end{cases}
$$

To confirm this is a proper pdf, check if the CDF = 1

$$
\begin{align*}
\int_{-\infty}^\infty f_X(x)dx &= \int_{5}^{15} \frac{1}{10}dx\\
&= \frac{1}{10}(x|^{15}_5)\\
&= \frac{1}{10}(15-5)\\
&= \frac{1}{10}(10)\\
&= \boxed{1}
\end{align*}
$$

Therefore the $f_X(x)$ is valid.

Next double check if the expected value equals to 10:

$$
\begin{align*}
E(X) &= \int_{-\infty}^\infty x \cdot f_X(x)dx\\
&= \int_{5}^{15}x \cdot \frac{1}{10}dx\\
&= \frac{1}{10} \int_{5}^{15}xdx\\
&= \frac{1}{10}(\frac{x^2}{2} |^{15}_{5})\\
&= \frac{1}{10}(\frac{15^2}{2} - \frac{5^2}{2})\\
&= \frac{1}{10}(\frac{225}{2} - \frac{25}{2})\\
&= \frac{1}{20}(200)\\
&= \frac{200}{20}\\
&= \boxed{10}
\end{align*}
$$

Therefore the given probability distribution function is accurate


---
## 8

In this question we’ll derive some intuition for the shape of the PDF for a Normal distribution $N(\mu, \sigma^2)$. To this end, consider the following functions

$$
g(x) = e^{-x^2/2}
$$

and

$$
h(x) =
\begin{cases}
1 + x & \text{if } -1 < x \le 0,\\
1 - x & \text{if } 0 < x \le 1,\\
0     & \text{otherwise.}
\end{cases}
$$

For the following questions you need to illustrate your answers. For all diagrams use the limits on the x-axis to range from $x \in [-5, 5]$. A hand-drawn sketch is acceptable as long as you label your drawings/sketches appropriately. You can also use the free online tools at Desmos or GeoGebra to plot these functions.

### (a) (5 points)

Sketch a plot of $y = h(x)$, and $y = g(x)$ side by side for $x \in [-5, 5]$. Is it symmetric? Where is it centered?

**SOLUTION**

![[Pasted image 20260201140744.png]]

The green is $g(x)$ and blue indicates $h(x)$
The graph is symmetric around the the $y$-axis ($x = 0$)

### (b) (5 points)

Recalling function transformations, what does the plot for $y = h(x - 1)$ look like? Where is this new plot centered? What effect does $y = h(x - 1)$ have in relation to $y = h(x)$?

**SOLUTION**

![[Pasted image 20260201141334.png]]

The green is there $y = h(x-1)$. 

From the graph is clear that it retains the same shape of $h(x)$ but transformed to the right by $1$. This new plot is centered at $x=1$

### (c) (5 points)

What is the expression for $g(x - 1)$? Based on the intuition from part (b), approximately sketch the graph for the function $y = g(x - 1)$.

**SOLUTION**

![[Pasted image 20260201141723.png]]

The purple is the when $y = g(x-1)$

$$
\begin{align*}
g(x-1) = e^{-\frac{(x-1)^2}{2}}
\end{align*}
$$
### (d) (5 points)

Recalling function transformations, what does the plot for $y = h(x/2)$ look like? Where is this new plot centered? What effect does $y = h(x/2)$ have in relation to $y = h(x)$?

**SOLUTION**

![[Pasted image 20260201142209.png]]

The black line indicates the $h(\frac{x}{2})$. This new plot is centered around $x = 0$. The expression $y = h(\frac{x}{2})$ effectively spread the original $h(x)$ to a wider triangle.

### (e) (5 points)

What is the expression for $g(x/2)$? Based on the intuition from part (d), approximately sketch the graph for the function $y = g(x/2)$.

**SOLUTION**

![[Pasted image 20260201142411.png]]

The new blue line indicates the function $g(\frac{x}{2})$

The expression for $g(\frac{x}{2})$ with be the following:

$$
g(\frac{x}{2}) = e^{-\frac{x^2}{8}}
$$
### (f) (5 points)

Based on the intuition developed so far, let $\mu \in \mathbb{R}$ and $\sigma \in \mathbb{R}$ be two fixed constants. Write down the expression for $y = g\big((x - \mu)/\sigma\big)$. Describe how the graph of the function $y = g\big((x - \mu)/\sigma\big)$ looks in relation to $y = g(x)$.

**SOLUTION**

The new $y = g(\frac{x-\mu}{\sigma})$ can be written as the following:

$$
\begin{align*}
g(\frac{(x-\mu)}{\sigma}) &= e^{-\frac{\frac{x-\mu}{\sigma}^2}{2}}\\
&= e^{-\frac{(x-\mu)^2}{2\sigma^2}}

\end{align*}
$$

Which based on the values of $\mu$ and $\sigma$, they have the following effect in the graph:
- $\mu$: transforms the graph left and right (right → +, left → -)
- $\sigma$: alters the thickness of the graph (the closer to 0, the thinner the graph shows)

### (g) (5 points)

Write down the pdf of the normal distribution with mean $\mu$ and variance $\sigma^2$, which we have seen in class, and denote this as $f(x)$. Thinking about function transformations, how will the plot of $y = f(x)$ differ from the plot of $y = g\big((x - \mu)/\sigma\big)$? Why do you think we include the leading coefficient in the pdf of the Normal distribution? Without using any calculus, based solely on the PDF of the normal distribution, what must be the area under the curve $y = f(x)$, taken over all $x \in \mathbb{R}$?

**SOLUTION**

For $X \sim N(\mu, \sigma^2)$, the probability distribution function is the following:

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

When comparing this to $g(\frac{x-\mu}{\sigma})$, the following is found:

$$
\begin{align*}
g(\frac{x-\mu}{\sigma}) = exp(-\frac{(x-\mu)^2}{2\sigma^2})\\
f(x) = \frac{1}{\sqrt{2\pi}\sigma}g(\frac{x-\mu}{\sigma})
\end{align*}
$$

With this establishment, the function $g$ has a multiplier in front, which contributes to the height by a factor of $\frac{1}{\sqrt{2\pi}\sigma}$, making the overall graph $g$ shorter in height
### (h) (5 points)

Given a random variable $X$ that is distributed normally with mean $\mu$ and variance $\sigma^2$, consider the random variable

$$
Z = \frac{X - \mu}{\sigma}.
$$

What does this transformation do to the mean and variance of $X$?

**SOLUTION**

Since the random variable $X \sim N(\mu, \sigma^2)$, the formula of expected value and variance should be from the normal distributions.

With $Z = \frac{X - \mu}{\sigma}$, finding the expected value and variance will give a better understanding of how the transformation effects the mean and variance

$$
E(Z) = \frac{E(X)}{\sigma} - \frac{\mu}{\sigma} = \frac{(\mu)}{\sigma} - \frac{\mu}{\sigma} = 0
$$

This suggests the transformation $Z$ shifted the mean from $\mu$ to $0$

$$
Var(Z) = (\frac{1}{\sigma})^2Var(X) = \frac{1}{\sigma^2} \sigma^2 = \boxed{1}
$$

This suggest the transformation $ZR$ rescaled the variance from $\sigma^2$ to $1$