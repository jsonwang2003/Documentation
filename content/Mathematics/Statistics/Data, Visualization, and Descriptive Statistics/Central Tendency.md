> [!ABSTRACT]
> 
> Central tendency refers to the statistical measures used to identify the "center" or "typical" value of a dataset. These metrics provide a summary that represents the entire distribution with a single point.

---
## 1. Primary Measures

### Mean
- The man of a set of quantitative variables is given by:

$$
\bar{x} = \boxed{\frac{1}{n}\sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \cdots x_n}{n}}
$$
 > [!Example] The mean of $(1, 2, 3, 4, 10, x)$ is $3.3333$. What is $x$?
 > > [!Info]- Answer
 > > To find $x$ we will go backwards on the operations in finding the mean $\bar{x}$
 > > $$
 > > \begin{align*}
 > > \bar{x} &= 3.3333\\
 > > n &= 6\\
 > > x &= \bar{x} \cdot n - (\sum_{i=1}^{n-1} x_i)\\
 > > x &= 3.3333 \cdot 6 - \sum_{i = 1} ^ {5} x_i\\
 > > &= 19.999... - (1 + 2 + 3 + 4 + 10)\\
 > > &= 20 - 20\\
 > > x &= \boxed{0}
 > > \end{align*}
 > > $$ 
### Median
- Let the data points be $x_1, x_2, ..., x_n$ arranged in **non-decreasing** order ($x_i \leq x_{i+1}$ for all $i$). Then the median, $M$, is:

$$
M = \begin{cases}
	X_{\frac{n+1}{2}} & \text{if } n \text{ is odd}\\
	\frac{X_{\frac{n}{2}} + X_{(\frac{n}{2} + 1)}}{2} & \text{if } n \text{ is even}
\end{cases}
$$
> [!Example] The median of $(1, 2, 3, 4, 10, x)$ is $2.5$. What is $x$?
> > [!INFO]- Answer
> > Since we find the median by taking the middle 2 values and take their average, we see $(2, 3, 4)$ to have the set that we look for, which the $\frac{2 + 3}{2} = 2.5$, therefore $2$ and $3$ should be the middle 2 values. 
> > With that in mind, it is required that $x \leq 2$ to satisfy this requirement
### Mode
- The mode of data points $x_1, x_2, \cdots, x_n$ is the value which appears most frequently

> [!Example] The mode of $(1, 2, 3, 4, 10, x)$ is $1$. What is $x$?
> > [!info]- Answer
> > Each known value in the data points are only themselves. In order for $1$ to be considered the most, the value $1$ must appear more than once. 
> > 
> > With that in mind, it is clear that $x =1$ to satisfy the condition

---

## 2. Comparing Measures by Distribution Shape

The relationship between the mean, median, and mode changes based on the **[[Data Visualization|Shape]]** of the distribution.

|**Distribution Shape**|**Relationship**|
|---|---|
|**Symmetric (Normal)**|Mean $\approx$ Median $\approx$ Mode|
|**Right-Skewed (Positive)**|Mode $<$ Median $<$ Mean|
|**Left-Skewed (Negative)**|Mean $<$ Median $<$ Mode|

---
## 3. When to Use Which?
- **Use Mean** when the data is symmetric and you need to account for every value in the set. 
- **Use Median** when you want to describe the "typical" experience in a skewed dataset (e.g., Household Income).
- **Use Mode** when you are dealing with non-numerical categories (e.g., "What is the most popular car color?").