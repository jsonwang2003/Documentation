- One of the key aspects of supervised machine learning is **model evaluation and validation**. 
- Two popular measures of evaluation

## Coefficient of Determination 

- Measures the **proportion of variance** in the target variable that is explained by the model
	- The best possible score for $R^2$ is 1.0 (when $u = 0$)
	- Lower values are worse
	- $R^2$ is 0.0 when $u = v$ 		
	 $$
		\begin{aligned}
		R^2 = (1 - \frac{u}{v})\\
		0 <= R^2 <= 1
		\end{aligned}
		$$
	- Where: 
		- the term $u$ is the **residual sum of squares**
			$$
			u = \sum (y - \hat y)^2
			$$
		- $y$ is the **observed response**
		- $\hat y$ is the **predicted response**
	- the term $v$ is the **total sum of squares**
		$$
		v = \sum (y - \bar y)^2
		$$
	- $\bar y$ is the **mean of the observed data**
## Mean Squared Error ($MSE$)

- Measures **average squared difference** between _actual_ and _predicted_ values
	- Lower MSE = better predictions

> [!IMPORTANT]
> It's an **absolute error metric** &rarr; the smaller, the better 

	$$
	MSE = \frac{1}{n} \sum (y - \hat y)^2
	$$
- Where:
	- the term $n$ is the **total number of observations** in the dataset
