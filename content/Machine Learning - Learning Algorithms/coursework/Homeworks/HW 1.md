Due: Oct 1st, 2026, 11:59pm (Pacific Time)
# 1 (10 points) Classification vs. Clustering
In this question, you are provided with several scenarios. You need to identify if the given scenario is better formulated as a classification task or a clustering task. You should also provide the reason that supports your choice.

## Scenario 1: 
Assume there are $100$ graded answer sheets for a homework assignment (scores range from 0 to 100). We would like to split them into several groups where each group has similar scores.

**Answer**
**Choice**: *Clustering* task
**Reason**: Since we do not know the explicit classes to classify individual homework assignments, so by performing clustering task, we can get the groups of all $100$ assignment scores in their relative position in the dataset.

## Scenario 2
Assume there are 100 graded answer sheets for a homework assignment (scores range from 0 to 100). We would like to split them into several groups where each group represents a letter grade (A, B, C, D) following the criteria: A (90-100), B (75-90), C (60-75), D (0-60).

**Answer**
**Choice**: *Classification* task
**Reason**: In this scenario, the classes to define individual scores have been defined. Therefore, the choice of classification is better suited for this scenario

# 2. (40 points) Basic Calculus
## 2.1 (20 points) Derivatives with Scalars
### 1. $f(x) = \frac{1}{2}(ax-b)^{2}$ where $a,b \in \mathbb{R}$ are constant scalars, derive $\frac{\partial f(x)}{\partial x}$

$$
	\begin{align*}
	\frac{\partial f(x)}{\partial x} &= \frac{\partial}{\partial x}\left( \frac{1}{2}(ax-b)^{2} \right)\\
	&= \frac{1}{2} \frac{\partial}{\partial x}(ax-b)^{2}\\
	&= \frac{1}{2} \frac{\partial}{\partial x}((ax)^{2} - 2abx + b^{2})\\
	&=
	\frac{1}{2} \frac{\partial}{\partial x}(ax)^{2} - \frac{\partial}{\partial x}(abx) + \frac{\partial}{\partial x} \frac{b^{2}}{2}\\
	&= a^{2}x - ab + 0\\
	&= \boxed{a^{2}x-ab}
	\end{align*}
$$

### 2. $f(x) = \ln(1+e^{x})$, derive $\frac{\partial f(x)}{\partial x}$
$$
\begin{align*}
\frac{\partial f(x)}{\partial x} &=\frac{\partial}{\partial x}(\ln (1+e^{x}))\\
&= \frac{1}{1+e^{x}} \frac{\partial}{\partial x}(1+e^{x})\\
&= \frac{1}{1+e^{x}} e^{x}\\
&= \boxed{\frac{e^{x}}{1+e^{x}}}
\end{align*}
$$

## 2.2 Derivatives with Vectors
Several particular vector derivatives are useful for this course. For matrix $A \in \mathbb{R}^{M\times M}$, column vector $x \in \mathbb{R}^{M}$ and $a \in \mathbb{R}^{M}$, we have
- $\frac{\partial a^{T}x}{\partial x} = \frac{\partial x^{T}a}{\partial x} = a$
- $\frac{\partial x^{T}Ax}{\partial x} = (A + A^{T})x$. if $A$ is symmetric, $\frac{\partial x^{T}Ax}{\partial x}=2Ax$
  A special case is, if $A=I$ (identity matrix), $\frac{\partial x^{T}x}{\partial x}=\frac{\partial x^{T}Ix}{\partial x}=2Ix = 2x$
### 1. $f(x) = \frac{1}{2}(x-a)^{\intercal}(x-a)$ where $a \in \mathbb{R}^{M}$ is a constant vector, derive $\frac{\partial f(x)}{\partial x}$
$$
	\begin{align*}
	\frac{\partial f(x)}{\partial x} &= \frac{\partial}{\partial x}\left[ \frac{1}{2} (x-a)^{\intercal}(x-a) \right]\\
	&= \frac{1}{2}\left[ \frac{\partial}{\partial x}(x-a)^{\intercal}(x-a) \right]\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left[ x^{\intercal}x - x^{\intercal}a - a^{\intercal}x + a^{\intercal}a \right]\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left[ x^{\intercal}x - a^{\intercal}x - a^{\intercal}x + a^{\intercal}a \right]\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left[ x^{\intercal}x - 2a^{\intercal}x + a^{\intercal}a \right]\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left( x^{\intercal}x \right) - \frac{\partial}{\partial x}\left( 2a^{\intercal}x \right) + \frac{\partial}{\partial x}\left( a^{\intercal}a \right)\\
	&= \frac{1}{2}[2x - 2a]\\
	&= \boxed{x-a}
	\end{align*}
$$

### 2. $f(x) = \frac{1}{2}(Ax-b)^{\intercal}(Ax-b)$ where $A \in \mathbb{R}^{M\times M}$ is a constant matrix and $b \in \mathbb{R}^{M}$ is a constant vector, derive $\frac{\partial f(x)}{\partial x}$
**Hint:** Note that $(A^{T}A)^{T} = A^{T}A$ is a symmetric matrix

$$
	\begin{align*}
	\frac{\partial f(x)}{\partial x} &= \frac{\partial}{\partial x}\left( \frac{1}{2}(Ax-b)^{\intercal}(Ax-b) \right)\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left[ \left( x^{\intercal}A^{\intercal}-b^{\intercal} \right)(Ax-b) \right]\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left[ x^{\intercal}A^{\intercal}Ax-x^{\intercal}A^{\intercal}b - b^{\intercal}Ax + b^{\intercal}b\right]\\
	&= \frac{1}{2} \frac{\partial}{\partial x}\left[ x^{\intercal}\left( A^{\intercal}A \right)x-2\left( A^{\intercal}b \right)^{\intercal}x + b^{\intercal}b \right]\\
	&=\frac{1}{2}\left[ \frac{\partial}{\partial x}x^{\intercal}\left( A^{\intercal}A \right)x - 2\frac{\partial}{\partial x}\left( A^{\intercal}b \right)^{\intercal} x + \frac{\partial}{\partial x}b^{\intercal}b\right]\\
	&= \frac{1}{2}\left[ 2A^{\intercal}Ax - 2A^{\intercal}b + 0 \right]\\
	&= A^{\intercal}Ax - A^{\intercal}b\\
	&= A^{\intercal}(Ax-b)
	\end{align*}
$$

# 3 (20 points) Metrics
In machine learning, we have many metrics to evaluate the performance of our model. For example, in a binary classification task, there is a dataset $S=\{ (x_{i}, y_{i}), i=1, \dots, N \}$ where each data point $(x,y)$ contains a feature vector $x \in \mathbb{R}^{M}$ and a ground-truth label $y\in \{ 0,1 \}$. We have obtained a classifier $f: \mathbb{R}^{M} \to \{ 0,1 \}$ to predict the label $\hat{y}$ of feature vector $x$:

$$
	\hat{y} = f(x)
$$
Assume $N=200$ and we have the following *confusion matrix* to represent the result of classifier $f$ on dataset $S$:

|                                     | Actual Positives ($y=1$) | Actual Negatives ($y=0$) |
| ----------------------------------- | ------------------------ | ------------------------ |
| Predicted Positives ($\hat{y} = 1$) | 5                        | 5                        |
| Predicted Negatives ($\hat{y}=0$)   | 10                       | 180                      |

## 1. Please Compute the *accuracy* of the classifier $f$ on dataset $S$

$$
\begin{align*}
	\text{accuracy} &= \frac{TP+TN}{TP+FP+FN+TN}\\
	&=	\frac{5+180}{5+5+10+180}\\
	&=\frac{185}{200} = \boxed{92.6\%}
\end{align*}
$$

## 2. Please compute the *precision* of the classifier $f$ on dataset $S$
$$
	\begin{align*}
	\text{precision} &= \frac{TP}{TP+FP}\\
	&= \frac{5}{5+5}\\
	&= \frac{5}{10} = \boxed{50\%}
	\end{align*}
$$

## 3. Please compute the *F1 score* of the classifier $f$ on dataset $S$

First find *Recall*
$$
	\begin{align*}
	\text{Recall} &= \frac{TP}{TP+FN}\\
	&= \frac{5}{5+10}\\
	&= \frac{5}{15} = \boxed{33\%}
	\end{align*}
$$
Then find *F1 score*
$$
	\begin{align*}
	\text{F1 score} &= \frac{2 \times \text{precision} \times \text{recall}}{\text{precision}+\text{recall}}\\
	&= \frac{2\times \frac{5}{10} \times \frac{5}{15}}{\frac{5}{10} + \frac{5}{15}}\\
	&= \frac{\frac{1}{3}}{\frac{25}{30}}\\
	&= \frac{10}{25} = \boxed{40\%}
	\end{align*}
$$
## 4. You may find the accuracy of current model very high. Does it mean the performance of this model is always very good? Why?
**Hint**: You may refer to other metrics you have computed

The high accuracy of this model is misleading due to the imbalance in the dataset $S$. The dataset is heavily leaning towards the negative class and only a small portion contributes to the positives. The other metrics appears to support this as the precision shows only a $50\%$ accuracy of positive predictions, recall showing a $33\%$ of identifying positive instances. Both of which resulted in the F1-score to score a mere $40\%$ accuracy of identifying the positive instances.

```python
plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.5)

plt.xlabel(wine.feature_names[0])

plt.ylabel(wine.feature_names[1])

plt.title("Scatter Plot for First 2 Feature Dimensions")

plt.show()
```

![[Pasted image 20260927172226.png]]

# 5. (20 points) Data Manipulation
We have already had a glimpse of the Wine dataset in Question 4. In this question, we will still use the Wine dataset. In fact, you can see the shape of array X is (178, 13) by running `X.shape`, which means it contains $178$ data points and $13$ features per data point. You may refer the the Jupyter notebook `HW1-Q4-Q5.ipynb` (https://tinyurl.com/fa26-cse151a-hw1-q5) for some skeleton code. Here, we will calculate some measures of the array X and perform some basic data manipulation: 
## 1. Show the first 2 features of the first 3 data points (i.e. first 2 columns and first 3 rows) of array X. (You can print the 3×2 array). 
```python
print(X[:3,:2])
```

```
[[14.23 1.71] 
[13.2 1.78] 
[13.16 2.36]]
```
## 2. Calculate the mean and the variance of the 1st feature (the 1st column) of array X. 

```python
print("Mean: ", np.mean(X[:,0]))
print("Variance: ", np.var(X[:,0]))
```

```
Mean: 13.00061797752809 
Variance: 0.6553597304633255
```
## 3. Randomly sample 3 data points (rows) of array X by randomly choosing the row indices. Show the indices and the sampled data points. Hint: You may use `np.random.randint()`. 

```python
random_indices = np.random.randint(0, X.shape[0], size=3)
sampled_points = X[random_indices]
  
print("Sampled Row Indices: ", random_indices)
print("Sampled Data Points: ")
print(sampled_points)
```

```
Sampled Row Indices: [ 70 124 129] 
Sampled Data Points: 
[[1.229e+01 1.610e+00 2.210e+00 2.040e+01 1.030e+02 1.100e+00 1.020e+00 3.700e-01 1.460e+00 3.050e+00 9.060e-01 1.820e+00 8.700e+02] [1.187e+01 4.310e+00 2.390e+00 2.100e+01 8.200e+01 2.860e+00 3.030e+00 2.100e-01 2.910e+00 2.800e+00 7.500e-01 3.640e+00 3.800e+02] [1.204e+01 4.300e+00 2.380e+00 2.200e+01 8.000e+01 2.100e+00 1.750e+00 4.200e-01 1.350e+00 2.600e+00 7.900e-01 2.570e+00 5.800e+02]]
```

## 4. Add one more feature (one more column) to the array X after the last feature. The values of the added feature for all data points are constant 1. Show the first data point (first row) of the new array. Hint: You may use `np.ones()` and `np.hstack()`. 

```python
column = np.ones((X.shape[0], 1))
X_new = np.hstack((X, column))

print("First Data Point:")
print(X_new[0])
```

```
First Data Point: 
[1.423e+01 1.710e+00 2.430e+00 1.560e+01 1.270e+02 2.800e+00 3.060e+00 2.800e-01 2.290e+00 5.640e+00 1.040e+00 3.920e+00 1.065e+03 1.000e+00]
```