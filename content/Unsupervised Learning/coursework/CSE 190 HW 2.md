## 1. Single linkage versus complete linkage.
Consider the following set of seven data points in $\mathbb{R}^{2}$: 
$$
A: \begin{bmatrix}
0 \\ 0
\end{bmatrix},
B: \begin{bmatrix}
1 \\ 1
\end{bmatrix}, 
C: \begin{bmatrix}
4 \\ 4
\end{bmatrix}, 
D: \begin{bmatrix}
6 \\ 1
\end{bmatrix},
E: \begin{bmatrix}
0 \\ 4
\end{bmatrix},
F: \begin{bmatrix}
6 \\ 4
\end{bmatrix},
G: \begin{bmatrix}
2 \\ 1
\end{bmatrix}
$$
We will use $l_{2}$ distance on this data.

### (a) Execute the single linkage algorithm by hand on this data set, and show (a scanned picture of) the resulting dendrogram (you don't need to show the intermediate steps). The leaves should be labeled A through G, and the heights of branches should reflect the order in which merges occur.

**Answer**
![[Pasted image 20261013095606.png]]

### (b) Do the same for complete linkage.

**Answer**
![[Pasted image 20261013095647.png]]
## 2. For this problem, we'll be using the animals with attributes data set. 
Load the real-valued array, and also the animal names, into Python. Now hierarchically cluster this data, using `scipy.cluster.hierarchy.linkage`. Choose Ward's method, and plot the resulting tree using the dendrogram method, setting the orientation parameter to 'right' and labeling each leaf with the corresponding animal name.

### (a) Show the dendrogram that you get.

**Answer**
![[dendrogram.png]]
### (b) Ward's method of average linkage is essentially trying to minimize the k-means cost function. Let's see how well it does. Take $k=10$ in what follows:

#### Show the k-clustering returned by Ward's method. 
(You can reconstruct this by processing the matrix returned by `scipy.cluster.hierarchy.linkage`, which spells out the sequence of mergers that produced the hierarchy.) What is its k-means cost?

**Answer**
Ward's method cost: 679136.27
Cluster  1 ( 7 animals): tiger, leopard, fox, wolf, weasel, bobcat, lion
Cluster  2 ( 6 animals): dalmatian, persian cat, german shepherd, siamese cat, chihuahua, collie
Cluster  3 ( 3 animals): grizzly bear, giant panda, polar bear
Cluster  4 ( 8 animals): skunk, mole, hamster, squirrel, rabbit, rat, mouse, raccoon
Cluster  5 ( 4 animals): spider monkey, gorilla, chimpanzee, bat
Cluster  6 ( 2 animals): beaver, otter
Cluster  7 ( 6 animals): killer whale, blue whale, humpback whale, seal, walrus, dolphin
Cluster  8 ( 3 animals): hippopotamus, elephant, rhinoceros
Cluster  9 ( 6 animals): moose, ox, sheep, buffalo, pig, cow
Cluster 10 ( 5 animals): antelope, horse, giraffe, zebra, deer

#### Run k-means on this data, 10 times (each time initializing with 10 centers chosen at random from the data). Pick out the best (lowest cost) solution and show it. What is its cost?

**Answer**
Best K-Means Cost(Run 7): 716943.85
Cluster  1 ( 2 animals): grizzly bear, polar bear
Cluster  2 ( 3 animals): spider monkey, gorilla, chimpanzee
Cluster  3 ( 4 animals): hippopotamus, elephant, rhinoceros, pig
Cluster  4 ( 7 animals): killer whale, blue whale, humpback whale, seal, otter, walrus, dolphin
Cluster  5 ( 7 animals): dalmatian, persian cat, siamese cat, skunk, chihuahua, giant panda, collie
Cluster  6 ( 1 animals): bat
Cluster  7 ( 7 animals): beaver, mole, hamster, squirrel, rabbit, mouse, raccoon
Cluster  8 (10 animals): antelope, horse, moose, ox, sheep, giraffe, buffalo, zebra, deer, cow
Cluster  9 ( 5 animals): german shepherd, tiger, leopard, wolf, lion
Cluster 10 ( 4 animals): fox, rat, weasel, bobcat
## 3. The k-center cost function. 
An alternative to k-means is the k-center clustering problem, defined as follows: 
- Input: Data points $x_{1},...,x_{n}$ in some metric space $(\chi, d)$; integer $k>0$. 
- Output: "Centers" $\mu_{1},...,\mu_{k}\in\mathcal{X}$. 
- Goal: Minimize 
$$
	\max_{i}\min_{1\leq j\leq k}d(x_{i}, \mu_{j})
$$

Whereas $k$-means minimizes the *total* (squared) distance from datapoints to their closest centers, this cost function minimizes the *largest* distance between a datapoint and its closest center.
The $k$-center cost function is NP-hard to optimize. However, there is a good heuristic for it known as *farthest-first traversal*
- Set $\mu_{1}$ to be any of the data points
- Repeat for $j=2, 3, \dots, k$:
	- Set $\mu_{j}$ to be the data point farthest from $\mu_{1}, \dots, \mu_{j-1}$

### (a) Consider the following data set in $\chi = \mathbb{R}^{2}$
$$
	\begin{bmatrix}
	1 \\ 1
	\end{bmatrix}, 
	\begin{bmatrix}
	1 \\ 2
	\end{bmatrix},
	\begin{bmatrix}
	1 \\ 8
	\end{bmatrix},
	\begin{bmatrix}
	2 \\ 1
	\end{bmatrix},
	\begin{bmatrix}
	2 \\ 2
	\end{bmatrix},
	\begin{bmatrix}
	5 \\ 1
	\end{bmatrix},
	\begin{bmatrix}
	6 \\ 1
	\end{bmatrix}
$$
What is the optimal k-center solution for this data set (using $l_{2}$ distance), for $k=3$? Remember that the centers can be arbitrary points in $\chi$. Give the optimal set of centers as well as the cost.

**Answer**
$$
	\begin{align*}
	\mu_{1} &=\begin{bmatrix}
	1 \\
	8
	\end{bmatrix}\\
	\mu_{2} &= \begin{bmatrix}
	1.5 \\
	1.5
	\end{bmatrix}\\
	\mu_{3} &= \begin{bmatrix}
	5.5 \\
	1
	\end{bmatrix}
	\end{align*}
$$

$$
\begin{align*}
	\text{Optimal Centers cost} &= \max_{i}\min_{1\leq j\leq k}d(x_{i}, \mu_{j})\\
	&= max(0, \sqrt{ (2-1.5)^{2} + (2-1.5)^{2} }, \sqrt{ (6-5.5)^{2} })\\
	&= max\left( 0, \frac{\sqrt{ 2 }}{2}, \frac{1}{2} \right)\\
	&= \boxed{\frac{\sqrt{ 2 }}{2}}
\end{align*}
$$


### (b) Suppose we run farthest-first traversal on this data set, starting with $\mu_{1} = \begin{bmatrix}1 \\  1\end{bmatrix}$. Which centers will it pick as $\mu_{2}$ and $\mu_{3}$? What is the k-center cost of this solution?

**Answer**
$$
	\mu_{2} = \begin{bmatrix}
	1 \\
	8
	\end{bmatrix} \qquad\mu_{3}=\begin{bmatrix}
	6 \\
	1
	\end{bmatrix}
$$
$$
\begin{align*}
	\text{cost} &= \max_{i}\min_{1\leq j\leq k}d(x_{i}, \mu_{j})\\
	&= max(0, 1, 1, \sqrt{ 2 }, 0, 1, 0)\\
	&= \boxed{\sqrt{ 2 }}
\end{align*}
$$
## 4. Comparing the $l_{1}$, $l_{2}$, and $l_{\infty}$ norms.

### (a) Of all points $x\in\mathbb{R}^{d}$ with $\vert{}\vert{}x\vert{}\vert{}_{\infty}=1$, which has the largest $l_{1}$ norm? The largest $l_{2}$ norm?

**Answer**
$$
	\begin{align*}
	\text{from } ||x||_{\infty} &\implies|x_{1}| = |x_{2}| = \dots = |x_{d}| = 1\\
	l_{1, \max}&= \sum_{i=1}^{d}1 = d\\
	l_{2, \max} &= \sqrt{ \sum_{i=1}^{d} 1^{2}} = \sqrt{ d }
	\end{align*}
$$

### (b) Of all points $x\in\mathbb{R}^{d}$ with $\vert{}\vert{}x\vert{}\vert{}_{2}=1$, which has the largest $l_{1}$ norm? The largest $l_{\infty}$ norm?

**Answer**

$$
	\begin{align*}
	\text{from } ||x||_{2} &\implies \left( \sum_{i=1}^{d}1 \cdot |x_{i}| \right)^{2} \leq \left( \sum_{i=1}^{d} 1^{2} \right)\left( \sum_{i=1}^{d}x_{i}^{2} \right) = d \cdot 1 = d\\
	\text{since } d \cdot x_{i}^{2} = 1 &\implies |x_{i}| = \frac{1}{\sqrt{ d }}\\
	l_{1, \max} &= \sum_{i=1}^{d} \frac{1}{\sqrt{ d }} = \frac{d}{\sqrt{ d }} = \sqrt{ d } \\ \\
	x^{2}_{k} \leq \sum_{i=1}^{d} x_{i}^{2} = 1 &\implies |x_{k}| \leq {1}\\
	l_{\infty, \max} &= \max_{i} |x_{i}| = 1
	\end{align*}
$$

## 5. Weighted $l_{2}$ norm. 
Let $w_{1},...,w_{d}>0$ be any positive numbers. Let $w=(w_{1},...,w_{d})$ and define
$$
	||x||_{w} = \sqrt{ \sum_{i=1}^{d}w_{i}x_{i}^{2} }
$$
a weighted version of the $l_{2}$ norm on $\mathbb{R}^{d}$. Sketch the region $\vert{}\vert{}x\vert{}\vert{}_{w}\le1$ (that is, the unit ball of the $\vert{}\vert{}\cdot\vert{}\vert{}_{w}$ norm) for $d=2$ and $w=\begin{bmatrix}1 \\  4\end{bmatrix}$.

**Answer**
![[Pasted image 20261015165952.png]]
## 6. Sparsity norm? 
For $x\in\mathbb{R}^{d}$, define $f(x)$ to be the number of non-zero entries of $x$. For instance, $f((1, 0, 2, 0, 0)) = 2$. Is this a norm? Either show that it is, or explain why it isn't.

**Answer**
The function $f(x)$ is not a norm as it violates the homogeneous property of a norm ($l(tx) = |t|l(x)$). Suppose $x=(1,0)$ and $t=2$. This gives $f(2x) = f((2,0)) = 1$ whereas $2f(x) = 2(1) = 2$. Since the two does not match, it violates such property and this no a norm.

## 7. The following table specifies a distance function on the space $\chi = \{ A,B,C,D \}$. Is this a metric? Justify your answer.


|     | A   | B   | C   | D   |
| --- | --- | --- | --- | --- |
| A   | 0   | 2   | 1   | 5   |
| B   | 2   | 0   | 4   | 3   |
| C   | 1   | 4   | 0   | 2   |
| D   | 5   | 3   | 2   | 0   |

**Answer**
This is not a metric due to one of the violation in property 4: $d(x,z) \leq d(x,y) + d(y,z)$. This distance function fails this property with the given scenario:

$$
	\begin{align*}
	d(B,C) &\leq d(B,A) + d(A,C)\\
	4 &\leq 2 + 1\\
	4 &\leq 3 \implies \text{False Statement, failed property}
	\end{align*}
$$
## 8. Probability Simplex
For any integer $m\geq {2}$, the probability simplex $\Delta_{m}$ is the set of all probability distributions over $m$ outcomes

$$
	\Delta_{m} = \left\{  (p_{1}, \dots, p_{m}): p_{i} \geq {0}, \sum_{i}p_{i} = 1  \right\}
$$

For instance, for $m=2$, this is the set of all pairs $(p, 1-p)$ with $0 \leq p \leq {1}$
Sketch the probability simplex $\Delta_{3}$ using three-dimensional axes. Make sure to label the axes. Show the locations of the following points: 

$$
	\begin{bmatrix}
	1 \\
	0 \\
	0
	\end{bmatrix},
	\begin{bmatrix}
	0 \\
	1 \\
	0
	\end{bmatrix},
	\begin{bmatrix}
	0 \\
	0 \\
	1
	\end{bmatrix}
$$
Also label the most central point in the simplex; what are its coordinates

**Answer**
![[Pasted image 20261014115758.png]]

## 9. Jaccard Example
What is the Jaccard similarity between the following two sets?
- $A=\{ 1,3,5,7,9 \}$
- $B = \{ 2,3,5,7 \}$

**Answer**
$$
	\begin{align*}
	|A \cap B| &= |\{ 3, 5, 7 \}| = 3\\
	|A \cup B| &= |\{ 1, 2, 3, 5, 7, 9 \}| = 6\\
	\\
	s(A,B)_{\text{Jaccard}} &= \frac{|A\cap B|}{|A \cup B|} = \frac{3}{6} = \boxed{\frac{1}{2}}
	\end{align*}
$$

## 10. Cosine Similarity
### (a) Compute the cosine similarity between $x=\begin{bmatrix}1 \\  2 \\  3\end{bmatrix}$ and $x' = \begin{bmatrix}3 \\  2 \\  1\end{bmatrix}$

**Answer**
$$
\begin{align*}
	s(x, x') &= \frac{x \cdot x'}{||x||\ ||x'||}\\
	&= \frac{3 + 2 \cdot 2 + 3}{\sqrt{ 1 + 4 + 9 }\cdot \sqrt{ 9+4+1 }}\\
	&= \frac{10}{14}\\
	&= \boxed{\frac{5}{7}}
\end{align*}
$$

### (b) When is the cosine similarity between two vectors equal to zero? Give a precise characterization in terms of the angle between the vectors.

**Answer**
For the cosine similarity to be 0, the dot product between the vector must be 0. Since the dot product can be determined by $x \cdot x' = ||x||||x'|| \cos(\theta)$ and $||x'||, ||x|| > 0$, the $\cos(\theta) = 0$ for the dot product to be 0. As such, the angle $\theta = \pm {90}$ degrees.

### (c) Suppose that data lie in $\mathbb{R}^{2}$. Sketch the set of points whose cosine similarity to $x = \begin{bmatrix}1 \\  2\end{bmatrix}$ is at least $0.9$. In your picture, mark $x$; the rest of the sketch can be very rough, as long as it gives approximately the correct *shape* of the region.

**Answer**
$$
	\cos ^{-1}(0.9) = 25.8 \implies \theta \in [-25.8, 25.8] \text{ in respect to } \vec{x}
$$

![[Pasted image 20261014122055.png]]