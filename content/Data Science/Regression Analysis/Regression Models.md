# Regression Models

> [!INFO]
> providing predictive insights across various industries

- selecting the right regression model depends on
	- data characteristics
	- business objectives
	- computational constraints

## Linear Regression

The technique for **creating linear models**
### Simple Linear Regression
- Considers $n$ samples of a single variable $x \in R^n$ and describes the relationship between the variable and the response with the model: 

	$$
	y = a_0 +a_1x
	$$

- The relationship between the variable and the response are described with a straight line. 
- The constant $a_0$ 
	- Necessary to **not force the model to pass through the origin $(0, 0)$**. 
	- Equals to the value at which **the line crosses the y-axis**. 
	- Not interpretable
		- For example, if we have the regression between 
			- $x = height$
			- $y = weight$
			- does not make sense to interpret the value of $y$ when the height is 0

### Multiple Linear Regression
The objective of performing a regression is to build a model to express the relation between the response $y \in R^n$ and a combination of one or more (independent) variables $x_i \in R^n$. 
- The model allows us to predict the response $y$ from the predictors. 
- The simplest model which can be considered is a linear model, where the response $y$ depends linearly on the $d$ predictors $x_i$:
	$$
		y = a_0 +a_1x_1 +···+a_dx_d
	$$

Where:
- $a_i$ = **parameters** = **coefficients of the model**
- $a_0$ = **intercept** = **the constant term** 

This equation can be rewritten in a more compact (matricial) form: **$y = Xw$**, where 
$$
y=\begin{pmatrix} y_1\\ y_2\\ \vdots\\ y_n \end{pmatrix}, \quad
X=\begin{pmatrix} X_{11} & \cdots & X_{1d} \\
                  X_{21} & \cdots & X_{2d} \\
                  \vdots & \ddots & \vdots \\
                  X_{n1} & \cdots & X_{nd} \end{pmatrix}, \quad
w=\begin{pmatrix} a_1\\ a_2\\ \vdots\\ a_n \end{pmatrix}
$$
In the matricial form we add a constant term by **changing the matrix $X$ to $(1, X)$**.

## Polynomial Regression, Ridge, and Lasso Regression

- enhance performance when dealing with non-linearity and multicollinearity
## ElasticNet

- balances feature selection and regularization
- ideal for high-dimensional data
## Logistic Regression

- fundamental classification tasks approach
## Multivariate Regression

- valuable when predicting multiple outcomes simultaneously
## SVR, Decision Trees, Random Forest

- provide sophisticated solutions for complex problems