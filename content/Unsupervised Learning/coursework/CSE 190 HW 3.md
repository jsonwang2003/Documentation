## 1. *Unit Vectors.* 
Find unit vectors in each of the following directions

### a) $\begin{bmatrix}3 \\  4\end{bmatrix} \in \mathbb{R}^{2}$

**Answer**
$$
	\begin{align*}
	u &= \begin{bmatrix}
	3 \\
	4
	\end{bmatrix}\\
	\hat{u} &= \frac{1}{\sqrt{ 3^{2}+4^{2} }}\begin{bmatrix}
	3 \\
	4
	\end{bmatrix}\\
	&= \frac{1}{5}\begin{bmatrix}
	3 \\
	4
	\end{bmatrix}\\
	&= \boxed{\begin{bmatrix}
	\frac{3}{5} \\
	\frac{4}{5}
	\end{bmatrix}}
	\end{align*}
$$

### b) $\begin{bmatrix}-4 \\  12 \\  3\end{bmatrix}\in \mathbb{R}^{3}$

**Answer**
$$
	\begin{align*}
	u &= \begin{bmatrix}
	-4 \\
	12 \\
	3
	\end{bmatrix}\\
	\hat{u} &= \frac{1}{\sqrt{ (-4)^{2}+12^{2}+3^{2} }}\begin{bmatrix}
	-4 \\
	12 \\
	3
	\end{bmatrix}\\
	&= \frac{1}{13}\begin{bmatrix}
	-4 \\
	12 \\
	3
	\end{bmatrix}\\
	&= \boxed{\begin{bmatrix}
	-\frac{4}{13} \\
	\frac{12}{13} \\
	\frac{3}{13}
	\end{bmatrix}}
	\end{align*}
$$

### c) The all-ones vector in $\mathbb{R}^{d}$

**Answer**
$$
\begin{align*}
	u &= \begin{bmatrix}
	1 \\
	1 \\
	\vdots \\
	1
	\end{bmatrix}\\
	\hat{u} &= \frac{1}{\sqrt{ 1^{2} + 1^{2} + \dots+1^{2} }}\begin{bmatrix}
	1 \\
	1 \\
	\vdots \\
	1
	\end{bmatrix}\\
	&= \frac{1}{\sqrt{ d }}\begin{bmatrix}
	1 \\
	1 \\
	\vdots \\
	1
	\end{bmatrix}\\
	&= \boxed{\begin{bmatrix}
	\frac{1}{\sqrt{ d }} \\
	\frac{1}{\sqrt{ d }} \\
	\vdots \\
	\frac{1}{\sqrt{ d }}
	\end{bmatrix}}
\end{align*}
$$

---
## 2.  
Let $x = \begin{bmatrix}-2 \\  4\end{bmatrix}$ and $u = \begin{bmatrix}1 \\  2\end{bmatrix}$

### a) What is the projection of point $x \in \mathbb{R}^{2}$ onto the direction of $u$?

**Answer**
$$
\begin{align*}
	\frac{x \cdot u}{||u||} &= \frac{1}{\sqrt{ 2^{2} + 1^{2} }}\sum_{i=1}^{d}u_{i}x_{i}\\
	&= \frac{1}{\sqrt{ 5 }}((1)(-2) + (2)(4))\\
	&= \frac{1}{\sqrt{ 5 }}(-2 + 8)\\
	&= \boxed{\frac{6}{\sqrt{ 5 }}}
\end{align*}
$$

### b) On a grid, plot point $x$, indicate direction $u$ and show the projection. (If you do this by hand, take a picture of it and include the picture in your homework)

**Answer**
![[Pasted image 20261022212938.png]]


---
## 3. 
Let $x = \begin{bmatrix}-4 \\  3\end{bmatrix} \in \mathbb{R}^{2}$

### a) Find the (unit) direction of $u$ such that the projection of $x$ onto $u$ is maximized

**Answer**
Since projection is the dot product of two vectors, the projection will be maximized when the $\cos \theta$ is maximized.

$$
	\begin{align*}
	\cos\theta &\in [-1,1]\\
	\cos\theta &= 1\\
	\theta &= \cos ^{-1}1\\
	&= 0 + 2n\pi, n \geq 0
	\end{align*}
$$
In other words, the angle between the vectors are in the exact same direction, as such, the unit vector that maximizes the projection equals $\hat{x}$

$$
\begin{align*}
	u = \hat{x} &= \frac{1}{\sqrt{ (-4)^{2} + (3)^{2} }}\begin{bmatrix}
	-4 \\
	3
	\end{bmatrix}\\
	&= \frac{1}{5}\begin{bmatrix}
	-4 \\
	3
	\end{bmatrix}\\
	&= \boxed{\begin{bmatrix}
	-\frac{4}{5} \\
	\frac{3}{5}
	\end{bmatrix}}
\end{align*}
$$

### b) Find the (unit) vector $u$ such that the projection of $x$ onto $u$ is minimized

**Answer**
Since projection is the dot product of two vectors, the projection will be minimized when the $\cos \theta$ is minimized.

$$
\begin{align*}
\cos\theta &\in [-1,1]\\
\cos\theta &= -1\\
\theta &= \cos ^{-1}(-1)\\
&= (2n+1)\pi, n \geq 0
\end{align*}
$$

In other words, the angle between the vectors are in the exact opposite direction, as such, the unit vector that minimizes the projection equals $-\hat{x}$

$$
	\begin{align*}
	u = -\hat{x} &= \frac{-1}{5}\begin{bmatrix}
	-4 \\
	3
	\end{bmatrix}\\
	&= \boxed{\begin{bmatrix}
	\frac{4}{5} \\
	-\frac{3}{5}
	\end{bmatrix}}
	\end{align*}
$$

### c) Find all (unit) directions $u$ such that the projection of $x$ onto $u$ is zero

**Answer**
Since projection is the dot product of two vectors, the projection will be zero when the $\cos \theta$ is zero.

$$
\begin{align*}
\cos\theta &\in [-1,1]\\
\cos\theta &= \\
\theta &= \cos ^{-1}(0)\\
&= \frac{\pi}{2}+n\pi, n \geq 0
\end{align*}
$$

In other words, the angle between the vectors are exactly $90^{o}$ from vector $x$, as such, the unit vector that zeros the projection have 2 options, $\pm {90}^{o}$

$$
\begin{align*}
u = \begin{cases}
\frac{1}{5}\begin{bmatrix}
-3 \\
-4
\end{bmatrix} &= \begin{bmatrix}
-\frac{3}{5} \\
-\frac{4}{5}
\end{bmatrix} &+90^{o} \\
\frac{1}{5}\begin{bmatrix}
3 \\
4
\end{bmatrix} &= \begin{bmatrix}
\frac{3}{5} \\
\frac{4}{5}
\end{bmatrix} &-90^{o}
\end{cases}
\end{align*}
$$

---
## 4. *The variance in different directions.*
A certain data set in $\mathbb{R}^{3}$ has covariance matrix

$$
	\Sigma = \begin{bmatrix}
	1 & 2 & 0 \\
	2 & 9 & 0 \\
	0 & 0 & 4
	\end{bmatrix}
$$

### a) What is the variance of the data in the direction $\begin{bmatrix}0 \\  1 \\  0\end{bmatrix}$


**Answer**
$$
\begin{align*}
	var(X) &= u^{T}\Sigma u\\
	&= \begin{bmatrix}
	0 \\
	1 \\
	0
	\end{bmatrix}^{T}
	\begin{bmatrix}
	1 & 2 & 0 \\
	2 & 9 & 0 \\
	0 & 0 & 4
	\end{bmatrix}
	\begin{bmatrix}
	0 \\
	1 \\
	0
	\end{bmatrix}\\
	&= \begin{bmatrix}
	0 & 1 & 0
	\end{bmatrix}
	\begin{bmatrix}
	2 \\
	9 \\
	0
	\end{bmatrix}\\
	&= 9
\end{align*}
$$

### b) What is the variance in the direction $\frac{1}{\sqrt{ 3 }}\begin{bmatrix}1 \\  1 \\  1\end{bmatrix}$

**Answer**
$$
\begin{align*}
var(X)&= u^{T}\Sigma u\\
&= \frac{1}{\sqrt{ 3 }}\begin{bmatrix} 
1 \\
1 \\
1
\end{bmatrix}^{T}
\begin{bmatrix}
1 & 2 & 0 \\
2 & 9 & 0 \\
0 & 0 & 4
\end{bmatrix}
\frac{1}{\sqrt{ 3 }}
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}\\
&= \frac{1}{3}\begin{bmatrix}
1  & 1 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 2 & 0 \\
2 & 9 & 0 \\
0 & 0 & 4
\end{bmatrix}
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}\\
&= \frac{1}{3}\begin{bmatrix}
1 & 1 & 1
\end{bmatrix}
\begin{bmatrix}
3 \\
11 \\
4
\end{bmatrix}\\
&= \frac{1}{3}(3+11+4)\\
&= \boxed{6}
\end{align*}
$$

---
## 5.
A particular data set in $\mathbb{R}^{5}$ happens to have a *diagonal* covariance matrix $\Sigma = diag(5, 10, 20, 1 ,4)$. What is the direction of maximum variance?

**Answer**
$$
	\Sigma = \begin{bmatrix}
	5 & 0 & 0 & 0 & 0 \\
	0 & 10 & 0 & 0 & 0 \\
	0 & 0 & 20 & 0 & 0  \\
	0 & 0 & 0 & 1 & 0 \\
	0 & 0 & 0 & 0 & 4
	\end{bmatrix}
$$
$var(X) = u^{T}\Sigma u$ is maximized by setting $u$ to the first eigenvector of $\Sigma$, in other words the eigenvector with the largest eigenvalue. As such, the first eigenvector = $\begin{bmatrix}0 \\  0 \\  1 \\  0 \\  0 \end{bmatrix}$ where $\lambda = 20$

---
## 6. 
A certain random variable $X \in \mathbb{R}^{3}$ has mean and covariance as follows:

$$
	\mathbb{E}X = \begin{bmatrix}
	1 \\
	2 \\
	3
	\end{bmatrix}, cov(X) = \begin{bmatrix}
	5 & -3 & 0 \\
	-3 & 5 & 0 \\
	0 & 0 & 4
	\end{bmatrix}
$$

### a) Consider the direction $u=\frac{1}{\sqrt{ 3 }}\begin{bmatrix}1 \\  1 \\  1\end{bmatrix}$. What are the mean and variance of $X \cdot u$

**Answer**
$$
\begin{align*}
	\mathbb{E}(X\cdot u) &= \mathbb{E}(X)\cdot u\\
	&= \begin{bmatrix}
	1 \\
	2 \\
	3
	\end{bmatrix}\cdot \frac{1}{\sqrt{ 3 }}\begin{bmatrix}
	1 \\
	1 \\
	1
	\end{bmatrix}\\
	&= \frac{1}{\sqrt{ 3 }}((1)(1)+(2)(1) + (3)(1))\\
	&= \frac{1}{\sqrt{ 3 }}(1+2+3)\\
	&= \frac{6}{\sqrt{ 3 }}\\
	&= \boxed{2\sqrt{ 3 }}\\ \\
	var(X\cdot u) &= u^{T}\Sigma u\\
	&= \frac{1}{\sqrt{ 3 }}\begin{bmatrix}
	1 & 1 & 1
	\end{bmatrix}
	\begin{bmatrix}
	5 & -3 & 0 \\
	-3  & 5 & 0 \\
	0 & 0 & 4
	\end{bmatrix}
	\frac{1}{\sqrt{ 3 }}\begin{bmatrix}
	1 \\
	1 \\
	1
	\end{bmatrix}\\
	&= \frac{1}{3}\begin{bmatrix}
	1 & 1 & 1
	\end{bmatrix}
	\begin{bmatrix}
	2 \\
	2 \\
	4
	\end{bmatrix}\\
	&= \frac{1}{3}(2+2+4)\\
	&= \frac{8}{3}
\end{align*}
$$

### b) The eigenvectors of $cov(X)$ can be found in the following list; which ones are they?
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
\begin{bmatrix}0 \\  0 \\  1\end{bmatrix},
\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
1 \\
1 \\
0
\end{bmatrix},
\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
0 \\
1 \\
1 
\end{bmatrix},
\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
1 \\
-1 \\
0
\end{bmatrix}
$$

**Answer**
$$
	\begin{align*}
	u &= \begin{bmatrix}
	1 \\
	0 \\
	0
	\end{bmatrix}
	&\Sigma u &= \begin{bmatrix}
	5 & -3  & 0
	\end{bmatrix} && No\\
	u &= \begin{bmatrix}
	0 \\
	1 \\
	0
	\end{bmatrix} &\Sigma u &= \begin{bmatrix}
	-3  & 5 & 0
	\end{bmatrix} &&No\\
	u &= \begin{bmatrix}
	0 \\
	0 \\
	1
	\end{bmatrix} &\Sigma u&=\begin{bmatrix}
	0 & 0 & 4
	\end{bmatrix} &&Yes, 4\cdot u\\
	u&=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
	1 \\
	1 \\
	0
	\end{bmatrix} &\Sigma u&=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
	2 & 2 & 0
	\end{bmatrix} &&Yes, 2\cdot u\\
	u&=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
	0 \\
	1 \\
	1
	\end{bmatrix} &\Sigma u&=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}-3 & 5 & 4
	\end{bmatrix} &&No\\
	u&=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
	1 \\
	-1 \\
	0
	\end{bmatrix}&\Sigma u&=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}
	8 & -8 & 0
	\end{bmatrix}&&Yes, 8\cdot u
	\end{align*}
$$

### c) Find the eigenvalues corresponding to each of the eigenvectors in part b. Make it clear which eigenvalue belongs to which eigenvector

**Answer**

| Eigenvector                                                          | Eigenvalue    |
| -------------------------------------------------------------------- | ------------- |
| $u = \begin{bmatrix}0 \\  0 \\  1\end{bmatrix}$                      | $\lambda = 4$ |
| $u=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}1 \\  1 \\  0\end{bmatrix}$    | $\lambda = 2$ |
| $u = \frac{1}{\sqrt{ 2 }}\begin{bmatrix}1 \\  -1 \\  0\end{bmatrix}$ | $\lambda = 8$ |

### d) Suppose we used principle component analysis (PCA) to project points $X$ into *two* dimensions. Which directions would it project onto?

**Answer**
Choose the $2$ eigenvectors with the largest eigenvalues $\implies u_{1}=\frac{1}{\sqrt{ 2 }}\begin{bmatrix}1 \\  -1 \\  0\end{bmatrix}, u_{2}=\begin{bmatrix}0 \\  0 \\  1\end{bmatrix}$

### e)
Continuing from part d, what would be the resulting two dimensional projection of the point $x = \begin{bmatrix}4 \\  0 \\  2\end{bmatrix}$

**Answer**

$$
\begin{align*}x'=
\begin{bmatrix}
	x\cdot u_{1} \\
	x\cdot u_{2}
\end{bmatrix}
	&= \begin{bmatrix}
	\frac{1}{\sqrt{ 2 }} & 0 \\
	-\frac{1}{\sqrt{ 2 }} & 0 \\
	0 & 1
	\end{bmatrix}^{T}\begin{bmatrix}
	4 \\
	0 \\
	2
\end{bmatrix}\\
&= \begin{bmatrix}
\frac{1}{\sqrt{ 2 }} & -\frac{1}{\sqrt{ 2 }} & 0 \\
0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
4 \\
0 \\
2
\end{bmatrix}\\
&= \begin{bmatrix}
2\sqrt{ 2 } \\
2
\end{bmatrix}
\end{align*}
$$

### f)
Continuing from part e, suppose that starting from the 2D projection, we tried to reconstruct the original $x$. What would the three-dimensional reconstruction be, exactly?

**Answer**
$$
	\begin{align*}
	x &= Ux'\\
	&= \begin{bmatrix}
	\frac{1}{\sqrt{ 2 }} & 0 \\
	-\frac{1}{\sqrt{ 2 }} & 0 \\
	0 & 1
	\end{bmatrix}\begin{bmatrix}
	2\sqrt{ 2 } \\
	2
	\end{bmatrix}\\
	&= \begin{bmatrix}
	2 \\
	-2 \\
	2
	\end{bmatrix}
	\end{align*}
$$

---
## 7.
$M$ is a $2 \times 2$ real-valued symmetric matrix with eigenvalues $\lambda_{1} = 2, \lambda_{2} = -1$ and corresponding eigenvectors

$$
	u_{1} = \frac{1}{\sqrt{ 5 }}\begin{bmatrix}
	2 \\
	1
	\end{bmatrix},
	u_{2}=\frac{1}{\sqrt{ 5 }}\begin{bmatrix}
	-1 \\
	2
	\end{bmatrix}
$$

### a)
What is $M$?

**Answer**
$$
	\begin{align*}
	M &= U\Lambda U^{T}\\
	&= \begin{bmatrix}
	\frac{2}{\sqrt{ 5 }}  & -\frac{1}{\sqrt{ 5 }}\\
	\frac{1}{\sqrt{ 5 }} & \frac{2}{\sqrt{ 5 }}
	\end{bmatrix}\begin{bmatrix}
	2 & 0 \\
	0 & -1
	\end{bmatrix}\begin{bmatrix}
	\frac{2}{\sqrt{ 5 }}& \frac{1}{\sqrt{ 5 }} \\
	-\frac{1}{\sqrt{ 5 }} & \frac{2}{\sqrt{ 5 }}
	\end{bmatrix}\\
	&= \begin{bmatrix}
	\frac{2}{\sqrt{ 5 }}  & -\frac{1}{\sqrt{ 5 }}\\
	\frac{1}{\sqrt{ 5 }} & \frac{2}{\sqrt{ 5 }}
	\end{bmatrix}\begin{bmatrix}
	\frac{4}{\sqrt{ 5 }} & \frac{2}{\sqrt{ 5 }} \\
	\frac{1}{\sqrt{ 5 }} & -\frac{2}{\sqrt{ 5 }}
	\end{bmatrix}\\
	&= \begin{bmatrix}
	\frac{8}{5} - \frac{1}{5} & \frac{4}{5} + \frac{2}{5} \\
	\frac{4}{5} + \frac{2}{5} & \frac{2}{5}-\frac{4}{5}
	\end{bmatrix}\\
	&= \begin{bmatrix}
	\frac{7}{5} & \frac{6}{5} \\
	\frac{6}{5} & -\frac{2}{5}
	\end{bmatrix}
	\end{align*}
$$

### b)
What are the eigenvalues of matrix $M + 2I$?

**Answer**
$$
\begin{align*}
(M+2I)u &= Mu + 2u = \lambda u + 2u = (\lambda+2)u\\
\\
\lambda_{1} &= 2 + 2 = 4\\
\lambda_{2}&= -1 + 2 = 1
\end{align*}
$$

### c)
What are the eigenvalues of the matrix $M^{2}  = MM$

**Answer**
$$
	\begin{align*}
	M^{2}u &= MMu = M(\lambda u) = \lambda(Mu) = \lambda(\lambda u) = \lambda^{2}u\\
	\\
	\lambda_{1} &= 2^{2} = 4\\
	\lambda_{2} &= (-1)^{2} = 1
	\end{align*}
$$

---
## 8.
The points $\begin{bmatrix}1 \\  1 \\  1\end{bmatrix}$ and $\begin{bmatrix}-1 \\  -1 \\  1\end{bmatrix}$ lie in a two-dimensional subspace of $\mathbb{R}^{3}$. What is the projection of $\begin{bmatrix}2 \\  4 \\  5\end{bmatrix}$ into this subspace? Your answer should be a vector in $\mathbb{R}^{3}$

**Answer**
1. Build the orthonormal basis
$$
	u_{1} = \frac{a-b}{||a-b||} = \frac{1}{\sqrt{ 2 }}\begin{bmatrix}
	1 \\
	1 \\
	0
	\end{bmatrix}\\
	u_{2} = \frac{a + b}{||a+b||} = \begin{bmatrix}
	0  \\
	0 \\
	1
	\end{bmatrix}
$$
2. Projection
$$
	\begin{align*}
	\begin{bmatrix}
	x\cdot u_{1} \\
	x\cdot u_{2}
	\end{bmatrix} &= \begin{bmatrix}
	\frac{1}{\sqrt{ 2 }} & \frac{1}{\sqrt{ 2 }} & 0 \\
	0 & 0 & 1
	\end{bmatrix}\begin{bmatrix}
	2 \\
	4 \\
	5
	\end{bmatrix}\\
	&= \begin{bmatrix}
	\frac{2}{\sqrt{ 2 }} + \frac{4}{\sqrt{ 2 }} \\
	5
	\end{bmatrix}\\
	&= \begin{bmatrix}
	3\sqrt{ 2 } \\
	5
	\end{bmatrix}
	\end{align*}
$$
3. Reconstruction
$$
	\begin{align*}
	(x\cdot u_{1})u_{1} + (x\cdot u_{2})u_{2}&= (3\sqrt{ 2 })\begin{bmatrix}
	\frac{1}{\sqrt{ 2 }} \\
	\frac{1}{\sqrt{ 2 }} \\
	0
	\end{bmatrix} + (5)\begin{bmatrix}
	0 \\
	0 \\
	1
	\end{bmatrix}\\
	&= \begin{bmatrix}
	3 \\
	3 \\
	0
	\end{bmatrix}+\begin{bmatrix}
	0 \\
	0 \\
	5
	\end{bmatrix}\\
	&= \begin{bmatrix}
	3 \\
	3 \\
	5
	\end{bmatrix}
	\end{align*}
$$

---
## 9. *Some Simple Visualization Methods*
For this problem, we will once again use the *animals with attributes* data set, in particular `predicate-matrix-continuous.txt`

### a)
We would like to visualize these animals in 2D. Do this with a PCA projection from $\mathbb{R}^{85} \to \mathbb{R}^{2}$. Show the position of each animal, and label each its name.
Python notes: You will need to make the plot larger by prefacing your code with
```python
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
```
(or try a different size if this doesn't seem right)

**Answer**
![[PCA Projecttion Graph.png]]

### b)
A popular visualization method is the *t-SNE* algorithm. This method takes a numerical parameter called the *perplexity* and then obtains an embedding by solving a non-convex optimization problem. 
The *t-SNE* algorithm is built into `scikit-learn`. You can invoke it to get a 2D embedding of a data set $X$ by using: 
```python
from sklearn.manifold import TSNE 
Z = TSNE(n components=2, perplexity=10.0).fit transform(X) 
```
The *perplexity* has a significant effect on the output. Try different perplexity values $(5,10,25,49)$ and show each of these embeddings, as you did with the PCA embedding. 
Bear in mind that *t-SNE* is a local search algorithm with randomized initialization, and thus even for a fixed perplexity value, different calls to it can return different results.

**Answer**
![[t-SNE Embedding p=5.png]]
![[t-SNE Embedding p=10.png]]![[t-SNE Embedding p=25.png]]
![[t-SNE Embedding p=49.png]]
### c)
How can we evaluate these embeddings? Some might seem more visually pleasing than others, or might seem to group animals in a way that agrees more with our own intuitions.
Let’s look at a somewhat more objective measure. Say we have a data set of $n$ points $x_{1},\dots x_{n}\in R^{d}$ and we somehow obtain a 2D visualization $z_{1}, \dots, z_{n}\in \mathbb{R}^{2}$ of them. How accurately does this 2D embedding capture the original interpoint distances? To see this
- Define $D_{ij} = ||x_{i} - x_{j}||$ and $D_{ij} = ||z_{i}-z_{j}||$.
- In general, the $D$ values might be scaled differently from the $\hat{D}$ values; so define the scaling factor to be $c = \frac{mean(D)}{mean(\hat{D})}$, where $mean(\cdot)$ denotes the average over all $n^{2}$ entries of the matrix. 
- The multiplicative factor by which the distance between $x_{i}$ and $x_{j}$ is distorted can be defined by
  $$
	\Delta_{ij} = \max\left( \frac{D_{ij}}{c \cdot \hat{D}_{ij}}, \frac{c \cdot \hat{D}_{ij}}{D_{ij}} \right)
  $$
  This ratio is always ≥ 1.
- The average distortion is then mean($\Delta$). 

Compute the average distortion for each of the five embeddings you found (PCA and four t-SNE embeddings). Which of them fares best under this measure?

**Answer**

--- Average Distortion Results ---
PCA:  1.8147
t-SNE (perplexity=5):  1.8980
t-SNE (perplexity=10):  1.5901
t-SNE (perplexity=25):  1.5672
t-SNE (perplexity=49):  1.7082

Based on these results, the best one under this measure is t-SNE at perplexity = 25 since its average distortion is the smallest amongst these methods