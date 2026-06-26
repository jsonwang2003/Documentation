# 1. Splitting the Data

- Dataset is divided into 2 or more subsets
	- Training set (70% - 80%)
		- used to fit the regression model by learning relationships between independent and dependent variables
	- Testing Set (20% - 30%)
		- used to evaluate the model's generalization performance on unseen data
		- helps assess whether the model is overfitting or underfitting
# 2. Model Training

- trained using training dataset by minimizing a loss function
	- Mean Squared Error (MSE)
	- Mean Absolute Error (MAE)
- steps:
	1. Selecting a regression algorithm (Linear Regression, Random Forest Regression, Support Vector Regression)
	2. Determining the best-fit parameters through methods like
		- gradient descent (linear models)
		- tree-based optimization (decision trees)

	3. Applying regularization techniques like these to prevent overfitting
		- Ridge (L2)
		- Lasso (L1)
# 3. Model Testing

- model is tested on the test database to measure its **performance**
	- ensures the model can make **accurate predictions** on unseen data
- Helps Identify:
	- Overfitting:
		- when the model performs ***well on training data but poorly on testing data***
	- Underfitting:
		- when the model ***fails to capture important patterns***
		- leading to **poor performance on both** training and test data