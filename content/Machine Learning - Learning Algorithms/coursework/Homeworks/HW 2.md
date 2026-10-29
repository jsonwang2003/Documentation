## 1 Nearest Neighbor Classification
You are given the points belonging to class-1 and class-2 as follows:
- Class 1 points: $(11,11),(13, 11), (8, 10), (9, 9), (7, 7), (7,5), (16,3)$
- Class 2 points: $(7, 11), (15,9), (15,7), (13,5), (14,4), (9,3), (11,3)$
What is the label of the sample $(14,3)$ using the nearest neighbor classifier using $L_{2}$ distance?

**Answer**
$$
	l_{2}((x_{1},y_{1}), (x_{2},y_{2})) = \sqrt{ (x_{1} - x_{2})^{2} + (y_{1}-y_{2})^{2} } 
$$
Distance to class 1:
- $(11,11) \implies \sqrt{ (14-11)^{2} + (3 - 11)^{2} } = \sqrt{ 9 + 64 } = \sqrt{ 73 } \approx {8}.5$
- $(13, 11) \implies \sqrt{ (14 - 13)^{2} + (3 - 11)^{2} } = \sqrt{ 1  +64 } = \sqrt{ (65) } \approx 8$
- $(8, 10) \implies \sqrt{ (14 - 8)^{2} + (3 - 10)^{2} } = \sqrt{ 36 + 49 } = \sqrt{ 85 } \approx 9.2$
- $(9, 9) \implies \sqrt{ (14-9)^{2} + (3 - 9)^{2} } = \sqrt{ 25 + 36 } = \sqrt{ 61 } \approx 7.8$
- $(7,7)\implies \sqrt{ (14 - 7)^{2} + (3 - 7)^{2} } = \sqrt{ 49 + 16 } = \sqrt{ 65 } \approx 8$
- $(7, 5) \implies \sqrt{ (14-7)^{2} + (3 - 5)^{2} } = \sqrt{ 49 + 4 } = \sqrt{ 53 } \approx 7.3$
- $(16, 3) \implies \sqrt{ (14-16)^{2} + (3-3)^{2} } = \sqrt{ 4 } = 2$
Minimum distance to class 1: $d_{\min, \text{Class 1}} = 2$

Distance to class 2:
- $(7,. 11) \implies \sqrt{ (14-7)^{2} + (3 - 11)^{2} } = \sqrt{ 49 + 64 } = \sqrt{ 113 } \approx 10.6$
- $(15, 9) \implies \sqrt{ (14-15)^{2} + (3-9)^{2} } = \sqrt{ 1 + 36 } = \sqrt{ (37) } \approx 6$
- $(15,7) \implies \sqrt{ (14-15)^{2} + (3-7)^{2} } = \sqrt{ 1 + 16 } = \sqrt{ (17) } \approx 4.1$
- $(13, 5) \implies \sqrt{ (14-15)^{2} + (3 - 5)^{2} } = \sqrt{ 1 + 4 } = \sqrt{ 5 } \approx 2.2$
- $(14,4)\implies \sqrt{ (14-14)^{2} + (3-4)^{2} } = \sqrt{ 1 } = 1$
- $(9,3)\implies \sqrt{ (14-9)^{2} + (3- 3)^{2} } = \sqrt{ 25 } = 5$
- $(11,3) \implies \sqrt{ (14-11)^{2} + (3-3)^{2} } = \sqrt{ (9) } = 3$
Minimum distance to class 2: $d_{\min, \text{Class 2}} = 1$

The nearest neighbor of $(14, 3)$ is the point $(14,4)$ which belongs to class 2. Thus the neighbor assigns $(14, 3)$ to class 2
## 2 Gradient Descent - Linear Regression
Consider house rent prediction problem where you are supposed to predict price of a house based on just its area. Suppose you have $n$ samples with their respective areas, $x^{(1)}, x^{(2)}, \dots, x^{(n)}$, their true house rents $y^{(1)}, y^{(2)}, \dots, y^{(n)}$. Let's say, you train a linear regressor that predicts $f(x^{(i)}) = \theta_{0} + \theta_{1}x^{(i)}$. The parameters $\theta_{0}$ and $\theta_{1}$ are scalars and are learned by minimizing mean-squared-error loss through gradient descent with a learning rate $\alpha$. Answer the following questions.

### (a) Express the loss function ($L$) in terms of $x^{(i)}, y^{(i)}, n, \theta_{0}, \theta_{1}$

**Answer**
$$
	L(\theta_{0}, \theta_{1}; x^{(i)}, y^{(i)}) = \frac{1}{2}\sum_{i=1}^{n}(f(x^{(i)}) - y^{(i)})^{2} = \frac{1}{2} \sum_{i=1}^{n}(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)})^{2}
$$

### (b) Compute $\frac{\partial L}{\partial\theta_{0}}$

**Answer**
$$
	\begin{align*}
	\frac{\partial L}{\partial\theta_{0}} &= \frac{\partial}{\partial \theta_{0}}\left( \frac{1}{2}\sum_{i=1}^{n}(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)})^{2} \right)\\
	&= \frac{1}{2} \sum_{i=1}^{n} 2(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)})\left( \frac{\partial}{\partial \theta_{0}}(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)}) \right)\\
	&= \frac{1}{2}\sum_{i=1}^{n}2(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)})(1)\\
	&= \sum_{i=1}^{n}\theta_{0}+\theta_{1}x^{(i)}-y^{(i)}
	\end{align*}
$$

### (c) Compute $\frac{\partial L}{\partial \theta_{1}}$

**Answer**
$$
	\begin{align*}
	\frac{\partial L}{\partial \theta_{1}} &= \frac{\partial}{\partial \theta_{1}}\left( \frac{1}{2}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)} - y^{(i)})^{2} \right)\\
	&= \frac{1}{2}\sum_{i=1}^{n}2(\theta_{0}+\theta_{1}x^{(i)} - y^{(i)})\left( \frac{\partial}{\partial \theta_{1}}(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)}) \right)\\
	&= \sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)} - y^{(i)})(x^{(i)})\\
	&= \sum_{i=1}^{n}(\theta_{0}x^{(i)} + \theta_{1}x^{2(i)} - x^{(i)}y^{(i)})
	\end{align*}
$$

### (d) Write update rules for $\theta_{0}$ and $\theta_{1}$

**Answer**
$$
	\begin{align*}
	\theta_{0} &\gets \theta_{0} - \alpha \frac{\partial L}{\partial \theta_{0}} = \theta_{0} - \alpha \sum_{i=1}^{n}(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)})\\
	\theta_{1} &\gets \theta_{1} - \alpha \frac{\partial L}{\partial \theta_{1}} = \theta_{1} - \alpha \sum_{i=1}^{n}(\theta_{0}x^{(i)} + \theta_{1}x^{2(i)} - x^{(i)}y^{(i)})	
	\end{align*}
$$
## 3 Gradient Descent - Linear Regression with L1 Regularization
Consider the same house rent prediction problem where you are supposed to predict price of a house based on just its area. Suppose you have $n$ samples with their respective areas, $x^{(1)}, x^{(2)}, \dots, x^{(n)}$, their true house rents $y^{(1)}, y^{(2)}, \dots, y^{(n)}$. Let's say, you train a linear regressor that predicts $f(x^{(i)}) = \theta_{0} + \theta_{1}x^{(i)}$. The parameters $\theta_{0}$ and $\theta_{1}$ are scalars and are learned by minimizing mean-squared-error loss with L1-regularization through gradient descent with a learning rate $\alpha$ and the regularization strength constant $\lambda$. Answer the following questions.

*Hint*: 

$$
	\frac{d|w|}{dw} = \begin{cases}
	1 &w > 0 \\
	\text{undefined}&w=0 \\
	-1 &w<0
	\end{cases}
$$
### (a) Express the loss function (L) in terms of $x^{(i)}, y^{(i)}, n, \theta_{0}, \theta_{1}, \lambda$

**Answer**
$$
	\begin{align*}
	L(\theta_{0}, \theta_{1}; x^{(i)}, y^{(i)}) = \frac{1}{2}\sum_{i=1}^{n}(f(x^{(i)}) - y^{(i)})^{2} + \lambda|\theta_{1}| = \frac{1}{2}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)} - y^{(i)})^{2} + \lambda||\theta_{1}||_{1}
	\end{align*}
$$
### (b) Compute $\frac{\partial L}{\partial \theta_{0}}$

**Answer**
$$
	\begin{align*}
	\frac{\partial L}{\partial \theta_{0}} &= \frac{\partial}{\partial \theta_{0}}\left( \frac{1}{2}\sum_{i=1}^{n}((\theta_{0}+\theta_{1}x^{(i)} - y^{(i)})^{2}) + \lambda|\theta_{1}| \right)\\
	&= \frac{1}{2}\sum_{i=1}^{n}2(\theta_{0}+\theta_{1}x^{(i)} - y^{(i)}) + 0\\
	&= \sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})
	\end{align*}
$$

### (c) Compute $\frac{\partial L}{\partial \theta_{1}}$

**Answer**
$$
	\begin{align*}
	\frac{\partial L}{\partial \theta_{1}} &= \frac{\partial}{\partial \theta_{1}}\left( \frac{1}{2}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})^{2} + \lambda|\theta_{1}| \right)\\
	&= \left( \frac{1}{2}\sum_{i=1}^{n}2(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)}) \frac{\partial}{\partial \theta_{1}}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})\right) + \frac{\partial}{\partial \theta_{1}}(\lambda|\theta_{1}|)\\
	&= \sum_{i=1}^{n}x^{(i)}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)}) + \lambda \ \frac{d|\theta_{1}|}{d\theta_{1}}
	\end{align*}
$$

### (d) Write update rules for $\theta_{0}$ and $\theta_{1}$

**Answer**
$$
	\begin{align*}
	\theta_{0}&\gets\theta_{0} - \alpha\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})\\
	\theta_{1}&\gets\theta_{1} - \alpha\left[\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})x^{(i)} + \lambda \frac{d|\theta_{1}|}{d\theta_{1}} \right]
	\end{align*}
$$
## 4 Gradient Descent - Linear Regression with L2 Regularization
Consider the same house rent prediction problem where you are supposed to predict price of a house based on just its area. Suppose you have $n$ samples with their respective areas, $x^{(1)}, x^{(2)}, \dots, x^{(n)}$, their true house rents $y^{(1)}, y^{(2)}, \dots, y^{(n)}$. Let's say, you train a linear regressor that predicts $f(x^{(i)}) = \theta_{0} + \theta_{1}x^{(i)}$. The parameters $\theta_{0}$ and $\theta_{1}$ are scalars and are learned by minimizing mean-squared-error loss with L2-regularization through gradient descent with a learning rate $\alpha$ and the regularization strength constant $\lambda$. Answer the following questions.

### (a) Express the loss function (L) in terms of $x^{(i)}, y^{(i)}, n, \theta_{0}, \theta_{1}, \lambda$

**Answer**
$$
	L(\theta_{0}, \theta_{1}) = \frac{1}{2}\sum_{i=1}^{n}(f(x^{(i)}) - y^{(i)})^{2} + \lambda||\theta_{1}||_{2} = \frac{1}{2}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)} - y^{(i)})^{2} + \lambda\theta_{1}^{2}
$$

### (b) Compute $\frac{\partial L}{\partial\theta_{0}}$

**Answer**
$$
\begin{align*}
	\frac{\partial L}{\partial\theta_{0}} &= \frac{\partial}{\partial\theta_{0}}\left( \frac{1}{2}\sum_{i=1}^{n}(\theta_{0} + \theta_{1}x^{(i)} - y^{(i)})^{2} + \lambda \theta_{1}^{2} \right)\\
	&= \sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})
\end{align*}
$$

### (c) Compute $\frac{\partial L}{\partial\theta_{1}}$

**Answer**
$$
	\begin{align*}
	\frac{\partial L}{\partial\theta_{1}} &= \frac{\partial}{\partial\theta_{1}}\left( \frac{1}{2}\sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})^{2} + \lambda\theta_{1}^{2} \right)\\
	&= \frac{1}{2}\sum_{i=1}^{n} 2(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})(x^{(i)}) + 2\lambda\theta_{1}\\
	&= \sum_{i=1}^{n}x^{(i)}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)}) + 2\lambda\theta_{1}
	\end{align*}
$$

### (d) Write update rules for $\theta_{0}$ and $\theta_{1}$

**Answer**
$$
	\begin{align*}
	\theta_{0} &\gets \theta_{0} - \alpha \sum_{i=1}^{n}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})\\
	\theta_{1}&\gets\theta_{1} - \alpha \sum_{i=1}^{n}(x^{(i)}(\theta_{0}+\theta_{1}x^{(i)}-y^{(i)})) + 2\lambda\theta_{1}
	\end{align*}
$$
## 5 Implementing a Linear Regression Model from Scratch
Now, you will implement a linear regression model from scratch. We have provided skeleton code files `LinearRegression.py` and `Linear_Regression.ipynb`) together with this home work, in the same Dropbox folder and Piazza post. The Python file (`LinearRegression.py`) is where you will implement the linear regression algorithm. The Jupyter Notebook file (`Linear_Regression.ipynb`) is where you will conduct experiments and answer relevant questions. Libraries such as `numpy` and `pandas` may be used for auxiliary tasks (such as matrix multiplication, matrix inversion, and so on), but not for the algorithms. That is, you can use `numpy` to implement your model, but cannot directly call libraries such as `scikit-learn` to get a linear regression model for your skeleton code. We will grade this question based on the three following criteria:

1. Your Implementation in code. Please do not change the structure of our skeleton code.
2. Your model's performance (we check if your model behaves correctly based on the results from multiple experiments in the notebook file)
3. Your written answers for questions in the notebook file

**LinearRegression.py**
```python
import numpy as np
from tqdm import tqdm
  
class Linear_Regression():
    def __init__(self, alpha = 1e-3 , num_iter = 10000, early_stop = 100, intercept = True, init_weight = None, penalty = None, lam = 1e-3, normalize = False, adaptive=True):
        """
        Linear Regression with gradient descent method.
  
        Attributes:
        -------------
        alpha: Learning rate.
        num_iter: Number of iterations    
        early_stop: Number of steps without improvements
                    that triggers a stop training signal
        intercept: True = with intercept (bias), False otherwise
        init_weight: Optional. The initial weights passed into the model,
                                for debugging
        penalty: {None,l1,l2}. Define regularization type for regression.
                  None: Linear Regression
                  l1: Lasso Regression
                  l2: Ridge Regression
        lam: regularization constant.
        normalize: True = normalize data, False otherwise
        adaptive: True = adaptive learning rate, False = fixed learning rate
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight
        self.penalty = penalty
        self.lam = lam
        self.normalize = normalize
        self.adaptive = adaptive
    
    def fit(self, X, y):
        # initialize X, y
        self.X = X
        self.y = y.reshape(-1, 1)
        self.max = np.zeros((1, X.shape[1]))
        self.min = np.zeros((1, X.shape[1]))
        for i in range(self.X.shape[1]):
            self.max[:, i] = self.X[:, i].max()
            self.min[:, i] = self.X[:, i].min()
  
        ############### START TODO 1 ###############
        if self.normalize:            
            self.X = (self.X - self.min) / (self.max - self.min)
        ############### END TODO 1 ###############
  
        ############### START TODO 2 ###############
        if self.intercept:
            self.X = np.hstack([self.X, np.ones((self.X.shape[0], 1))])
        ############### END TODO 2 ###############
  
        # initialize coefficient
        self.coef = self.init_weight if self.init_weight is not None\
        else np.array([np.random.uniform(-1,1,self.X.shape[1])]).T
  
        # start training, self.loss is used to record losses over iterations
        self.loss = []
        self.gradient_descent()
        
    def gradient(self):
        coef = -2 / len(self.X)
  
        ############### START TODO 3 ###############
        y_hat = self.X @ self.coef
        grad = coef * (self.X.T @ (self.y - y_hat))
        ############### END TODO 3 ###############
  
        ############### START TODO 4 ###############
        if self.penalty == 'l2':
            grad += 2 * self.lam * self.coef
        elif self.penalty == 'l1':
            grad += self.lam * np.sign(self.coef)
        else:
            pass
        ############### END TODO 4 ###############
  
        return grad
        
    def gradient_descent(self):
        print('Start Training')
        for i in range(self.num_iter):
  
            ############### START TODO 5 ###############
            previous_y_hat = self.X @ self.coef
            grad = self.gradient()
            temp_coef = self.coef - self.alpha * grad
            ############### END TODO 5 ###############
  
            ############### START TODO 6 ###############
            if self.penalty == 'l2':
                previous_reg_cost = self.lam * np.sum(np.square(self.coef))
                current_reg_cost = self.lam * np.sum(np.square(temp_coef))
            elif self.penalty == 'l1':
                previous_reg_cost = self.lam * np.sum(np.abs(self.coef))
                current_reg_cost = self.lam * np.sum(np.abs(temp_coef))
            else:
                previous_reg_cost = 0
                current_reg_cost = 0
            ############### END TODO 6 ###############
  
            ############### START TODO 7 ###############
            pre_error = np.sum(np.square(self.y - previous_y_hat)) + previous_reg_cost
            current_error = np.sum(np.square(self.y - self.X @ temp_coef)) + current_reg_cost
            ############### END TODO 7 ###############
  
            # Early Stop
            if len(self.loss) > self.early_stop and \
            self.loss[-1] >= max(self.loss[-self.early_stop:]):
                print('-----------------------')
                print(f'End Training (Early Stopped at iteration {i})')
                return self
            ############### START TODO 8 ###############
            if current_error < pre_error:
                self.alpha = self.alpha * 1.3 if self.adaptive else self.alpha
                self.coef = temp_coef
            else:
                self.alpha = self.alpha * 0.9 if self.adaptive else self.alpha
            ############### END TODO 8 ###############
            # record stats
            self.loss.append(float(current_error))
            if i % 1000000 == 0:
                print('-----------------------')
                print('Iteration: ' +  str(i))
                print('Coef: '+ str(self.coef))
                print('Loss: ' + str(current_error))
        print('-----------------------')            
        print('End Training')
        return self

    def predict(self, X):
        X_norm = np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_norm[:, i] = (X[:, i] - self.min[:, i]) / (self.max[:, i] - self.min[:, i])
        X = X_norm

        ############### START TODO 9 ###############
        if self.intercept:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
  
        y = X @ self.coef
        return y
        ############### END TODO 9 ###############
```

![[Linear_Regression.pdf]]