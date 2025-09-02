# # Logistic Regression

- widely used statistical method for **binary classification problems**
	- the outcome variable is categorical and typically represents two classes
- estimates the **probability that a given input belongs to a particular class**
	- predictions to a range between **0 and 1**
	- allow probability-based decision-making
- trained using **maximum likelihood estimation**
	- adjusting coefficients to minimize the difference between predicted and actual values

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset creation
data = {"MonthlyCharges": [29, 49, 69, 89, 109, 129, 149, 169, 189, 209],
		"Tenure": [1, 2, 5, 8, 12, 18, 24, 36, 48, 60],
		"Churn": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # 1 = Churn, 0 = No Churn
}

df = pd.DataFrame(data)

# Feature selection
X = df[['MonthlyCharges', 'Tenure']]
y = df['Churn']

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
```
### Output
- provides accuracy score indicating how well it predicts the data
- confusion matrix shows the number of **true positives, true negatives, false positives, false negatives**
- classification report provides essential metrics for **evaluating classification performance**
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
### Advantages

- interpretability
	- coefficients of the model indicate how each independent variable influences the likelihood of an outcome
	- performs well when the relationship between predictors and target variable is near linear in the log-odds space
- computationally efficient
	- requiring less training time than more complex models
### Disadvantages

- assumes linearity between independent and dependent variables
	- may not always hold true
- struggles with high-dimensional data
	- unless combined with **feature selection** or **regularization techniques like L1 (Lasso) and L2 (Ridge)**
- less effective in handling highly complex, non-linear relationship between variables
	- more sophisticated models may yield better results

# Classification Decision Trees

- powerful supervised learning technique used in machine learning for predictive modeling
- work by **recursively partitioning the dataset into subsets based on feature values, creating a tree-like structure**
	- each external node represents a decision rule
	- branches denote possible feature values
	- leaf nodes indicate the predicted class
- primary algorithm for constructing classification trees is the CART (Classification and Regression Trees) algorithm
	- measures **Gini impurity** or **entropy**
	- determines the best splits at each node
- useful for handling **categorical** and **numerical data**
	- offer intuitive representation of decision-making process

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset: Customer churn prediction
data = {'MonthlyBill': [50, 20, 80, 30, 90, 40, 70, 60, 25, 85],
		'ContractType': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 1: Long-term, 0: Short-term
		'CallDrops': [2, 5, 1, 6, 0, 7, 1, 3, 5, 1],
		'Churn': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # 1: Churn, 0: No churn
}

# Create DataFrame
df = pd.DataFrame(data)

# Split data into features and target variable
X = df[['MonthlyBill', 'ContractType', 'CallDrops']]
y = df['Churn']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Plot the decision tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=['MonthlyBill', 'ContractType', 'CallDrops'], 
		  class_names=['No Churn', 'Churn'], filled=True)
plt.show()
```
### Output

- providing insights through to understand how well the model is performing
	- Accuracy score: high score = model effectively identifies patterns based on dependent variables
	- classification report contains these for both churn and non-churn predictions
		- [[Model Evaluation#Precision|Precision]]
		- [[Model Evaluation#Recall|Recall]]
		- [[Model Evaluation#F1-Score|F1-Score]]
### Advantages

- highly interpretable
	- excellent choice when explainability is crucial
	- allows stakeholders to understand how and why the decisions are made
- can handle both numerical and categorical data
- require minimal preprocessing
- can capture complex relationships between features without requiring explicit feature transformations
### Disadvantages

- prone to overfitting
	- especially when tree is too deep
	- can capture noise instead of meaningful patterns
	- techniques like these mitigate this effect
		- pruning
		- setting a maximum depth
		- using ensemble methods like random forests
	- sensitive to small variations in data
		- small change can lead to significantly different tree structure
	- tend to bias toward dominant classes in imbalanced datasets
		- may require resampling or weighting techniques

# Random Forest Classification

- ensemble learning technique used for **classification** and **regression tasks**
- operates by **constructing multiple decision trees during training and outputs the mode of the individual tree predictions in classification problems**
	- majority vote
- helps mitigate overfitting by leveraging the **power of multiple models to improve generalization**
- algorithm introduced **randomness** by **selecting random subsets of features and samples to train each tree**
	- ensure diversity in decision boundaries
	- enhances accuracy, robustness, and stability

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'tenure': np.random.randint(1, 60, 500),
    'monthly_charges': np.random.uniform(20, 100, 500),
    'total_charges': np.random.uniform(100, 6000, 500),
    'customer_support_calls': np.random.randint(0, 10, 500),
    'contract_type': np.random.choice([0, 1], size=500),  # 0: Month-to-Month, 1: Long-Term
    'churn': np.random.choice([0, 1], size=500)  # 0: No churn, 1: Churn
})

# Split data into training and testing sets
X = data.drop(columns=['churn'])
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)<br><br>rf_classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Feature importance
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Plot feature importance
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()
```

### Output
- outputs an accuracy score
	- representing the proportion of correctly classified cases in the test set
- key metrics for evaluating model performance across different classes
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
- confusion matrix reveals true positive, true negative, false positive, and false negative
	- helps in assessing misclassification trends
- feature importance plot
	- provides insights into **which variables most significantly influence the classification outcome**
### Advantages

- able to handle large datasets with **high-dimensional features while maintaining robust performance**
- less prone to overfitting
- can handle both categorical and numerical data
	- versatile across different industries and problem domains
- interpretability through feature importance analysis
### Disadvantages

- computational complexity
	- require significant processing power and memory
	- making real-time prediction slower
- may struggle with imbalanced datasets
	- requiring techniques such as these to address bias in classification result
		- class weighting
		- resampling

# Support Vector Machines (SVM)

- powerful supervised learning algorithm used primarily for classification tasks but also applicable to regression problems
- work by **identifying the optimal hyperplane that best separates datapoints into distinct classes**
	- seeks to maximize the margin between data points and the decision boundary
	- helps improve generalization and reduces overfitting
- if data is not linearly separable
	- employ a kernel trick
	- transforming the feature space into a higher dimension where a linear separation becomes possible
	- Common kernels which all offers flexibility in handling different types of data distributions
		- linear
		- polynomial
		- radial bases function (RBF)
		- sigmoid
- robust to **high-dimensional** data and have strong theoretical underpinnings in statistical learning

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Generating synthetic data
np.random.seed(42)
num_samples = 500

# Creating two classes: Legitimate (0) and Fraudulent (1)
X_legit = np.random.normal(loc=50, scale=10, size=(num_samples // 2, 2))
X_fraud = np.random.normal(loc=70, scale=10, size=(num_samples // 2, 2))
X = np.vstack((X_legit, X_fraud))
y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training SVM with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

># Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Plot decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Classification of Fraudulent Transactions")
plt.show()

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
```

### Output

- confufion matrix provide insight into the model's performance by
	- indicating the number of true positives, true negatives, false positives, and false negatives
- classification report includes
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
### Advantages

- effectiveness in high-dimensional spaces, suitable for
	- text classification
	- image recognition
	- fraud detection
- excels in handling small to medium-sized datasets with complex decision boundaries
	-  kernel trick
- less prone to overfitting compared to some models
### Disadvantages

- computational efficiency when dealing with large datasets
	- training complexity scales poorly with the number of samples
- requires careful tuning of hyperparameters, poor parameters can lead to suboptimal decision boundaries
	- kernel type
	- regularization parameter (C)
	- gamma for RBF kernels
- does not provide probabilistic outputs directly
	- disadvantage where confidence scores are needed for decision-making

# Naïve Bayes

- probabilistic classification algorithm based on **Bayes' Theorem**

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

>[!INFO]
>calculates the posterior probability of a class given a set of features by **multiplying the prior probability of the class with the likelihood of the features occurring within that class**normalized by the probability of the features across all classes

- assuming the presence of particular feature in a class is independent of the presence of any other feature (conditional independence)
	- simplifies the computation and making it more efficient

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

# Split dataset into features and target variable
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naïve Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```
    
### Output

- accuracy score
- classification report
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
- precision and recall must be examined individually to assess performance on each class
### Advantages

- computationally efficient
	- doesn't require iterative parameter tuning
- performs well with categorical and text data
	- ideal for spam detection, sentiment analysis, and recommendation systems
- handles missing values well
	- probabilities are estimated independently for each feature
### Disadvantages

- assumption of feature independence often unrealistic
	- can lead to suboptimal classification performance
- struggles with small datasets where probability estimates may not be reliable
- not suitable for complex  relationships between features

# k-Nearest Neighbors (k-NN)

- simple but effective non-parametric machine learning technique used for both [[Summary of Supervised Learning#Classification|Classification]] and [[data-science/Regression Analysis/index|Regression]] tasks
- operates on the principle of similarity
	- observation is assigned to a class (or given a numerical value) based on the **majority / average of its k-nearest neighbors in the feature space**
- does not build an explicit model
	- stores the entire training dataset and makes predictions by measuring distances
		- typically Euclidean distance, but sometimes uses Manhattan or Minkowski distances as well
- choice of k (the number of neighbors considered) significantly impacts performance
	- smaller values lead to more sensitive models that capture local variations
	- larger values smooth the decision boundary but risk misclassifying minority cases

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Sample dataset: Customer age, income, and spending category (0 = Low, 1 = High)
data = {
    'Age': [25, 34, 45, 52, 23, 40, 60, 48, 33, 29],
    'Income': [40000, 60000, 80000, 75000, 30000, 72000, 95000, 88000, 54000, 50000],
    'Spending_Category': [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]  # Target variable<br><br>}

# Convert to DataFrame
df = pd.DataFrame(data)
# Splitting features and target
X = df[['Age', 'Income']]
y = df['Spending_Category']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying k-NN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
```
### Output

- classification report
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
- accuracy score
	- gives an overall measure of how well the model classifies the data
	- tuning k, adjust feature scaling, and increasing dataset size can optimize the performance
### Advantages

- simple
- effective with smaller datasets
- highly interpretable
	- does not require extensive training
	- naturally adapts to non-linear decision boundaries
### Disadvantages

- computationally expensive
	- especially for large datasets
	- every prediction requires calculating distances to all training points
- curse of dimensionality
	- performance degrades as the number of features increases
- sensitive to noisy and imbalanced data
	- few outliers or unequal class distributions can heavily impact predictions
- often mitigate by
	- optimize k
	- use feature selection to reduce dimensionality
	- apply distance weighting to improve decision accuracy

# Gradient Boosting Machines (GBM)

- powerful ensemble learning technique designed for predictive modeling tasks
	- particularly classification and regression
- belongs to the family of boosting algorithms
	- weak learners are sequentially trained in an additive manner to correct the mistakes of the previous iterations
- minimizes a differentiable loss function using gradient descent
	- adjusting each tree's contribution based on the negative gradient of the loss function
	- resulting a highly optimized predictive model capable of capturing complex relationships within data
- in classification context
	- aims to separate distinct classes by iteratively improving decision boundaries
	- starts from a simple model and refines it by focusing on the errors made in previous iterations
	- by learning from the residuals
		- continuously reduces classification errors
- excellent for requiring high predictive accuracy

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Generate synthetic dataset (for demonstration)
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    'MonthlyCharges': np.random.uniform(20, 100, data_size),
    'Tenure': np.random.randint(1, 72, data_size),
    'SupportCalls': np.random.randint(0, 10, data_size),
    'TotalUsage': np.random.uniform(100, 5000, data_size),
    'Churn': np.random.choice([0, 1], size=data_size, p=[0.8, 0.2])
})

# Splitting the dataset
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train GBM model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
```
### Output

```powershell
Model Accuracy: 0.87
Classification Report:
              precision    recall  f1-score   support
		0          0.90      0.95      0.92       160
		1          0.75      0.62      0.68        40
accuracy                               0.87       200
   macro avg       0.82      0.78      0.80       200
weighted avg       0.86      0.87      0.86       200
```

- based on the sample output
	- model achieves an accuracy of 87%
	- precision of 90% for non-churn customers
		- when model predicts the customer will not churn, the model is 90% correct of the time
	- recall of churned customers (62%)
		- model misses some actual churn cases
		- could be improved through hyperparameter tuning or additional feature engineering
### Advantages

- high predictive accuracy by iteratively refining the model through gradient-based optimization
- naturally handles complex, non-linear relationships between features
	- effective in real-world applications
- can handle missing data effectively
	- reducing the need for extensive preprocessing
### Disadvantages

- computational cost
	- sequential learning process can be time-intensive, especially with large datasets
- training process is inherently sequential
	- slower training speed
- prone to overfitting if not carefully tuned
	- especially with a higher number of trees or overly complex structures
	- proper hyperparameter tuning are necessary to mitigate this risk
		- adjusting learning rates
		- tree depths
- interpretability is a challenge
	- understanding its decision-making process is less intuitive

# Adaptive Boosting (AdaBoost)

- ensemble learning technique that **combines multiple weak classifiers to create a strong classifier**
- works iteratively by **training a sequence of base models** where each subsequent model focuses more on the **misclassified instances** from the previous iterations
	- decision trees with a single split (stumps)
- key idea is to assign weights to observations
	- increasing the importance of misclassified instances and adjust the model accordingly
- process continues until a predefined number of weak classifiers are trained, or the model achieves a perfect classification on the training data
- final prediction is made by aggregating the weighted votes of all weak classifiers

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AdaBoost classifier with decision stumps as base learners
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train the model
adaboost.fit(X_train, y_train)

# Make predictions
y_pred = adaboost.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Classification Accuracy: {accuracy:.4f}')
print('Classification Report:\n', report)
```
### Output

- classification accuracy
	- provides a measure of how well the model predicts the correct class labels
	- high accuracy score (85+ %): model is effective at distinguishing between the classes
- classification report for each class
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
### Advantages

- able to improve weak learners
	- highly effective for classification problems where traditional models might struggle
- resistant to overfitting
	- particularly when using small decision stumps
	- often outperforms standalone decision trees or logistic regression
- relatively easy to implement
- works well with various types of data
### Disadvantages

- can be sensitive to noisy data and outliers
	- assigns higher weights to misclassified instances
	- can amplify errors rather than correcting them
	- can lead to decreased model stability in datasets with high variance
- not well suited for handling extremely large datasets
	- computationally expensive due to its iterative nature
- less flexible in capturing complex patterns in data

# Stochastic Gradient Descent Classifier (SGD)

- optimization algorithm used in machine learning for efficiently training models
	- particularly those involving large-scale datasets
- linear classifier that utilizes the [[stochastic gradient descent optimization]] method to update model parameters iteratively
- updates the model parameters using a single sample per iteration
	- more computationally efficient and suitable for large datasets
- often used with linear models for classification tasks
- can incorporate different loss functions
	- hinge loss ([[Classification Models#Support Vector Machines (SVM)|Support Vector Machine]])
	- log loss ([[Classification Models#Logistic Regression|Logistic Regression]])

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generating a synthetic dataset (simulating e-commerce browsing behavior)
np.random.seed(42)
n_samples = 1000

# Features: products viewed, time spent (minutes), cart added (binary)
X = np.random.rand(n_samples, 3) * [10, 50, 1]  # Scaling different ranges
y = np.random.choice([0, 1], size=n_samples)  # Binary target: 1 (Purchase), 0 (No Purchase)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the SGD Classifier
sgd_clf = SGDClassifier(loss='log', max_iter=1000, learning_rate='optimal', random_state=42)
sgd_clf.fit(X_train, y_train)

# Predictions
y_pred = sgd_clf.predict(X_test)

# Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
```

### Outputs

- key performance metrics
	- including accuracy
	- measures the proportion of correctly predicted customer purchase decisions over the total number of test samples
	- high accuracy score &rarr; the model is effective in distinguishing between users likely to purchase and those who will not
- [[Model Evaluation#Precision|Precision]]
	- indicates  the percentage of predicted purchases that were actually purchases
	- $accuracy >= 80\%$:
		- SGD Classifier is performing well in predicting purchasing behavior
- [[Model Evaluation#Recall|Recall]]
	- how many actual purchases were correctly identified
- [[Model Evaluation#F1-Score|F1-Score]]
	- balanced measure of precision and recall
	- useful for assessing the model's ability to capture positive cases effectively
- if [[Model Evaluation#Recall|Recall]] or [[Model Evaluation#Precision|Precision]] score are low:
	- suggests issues where one class dominates the dataset
	- techniques like these may be necessary to improve performance and ensure the model generalizes well to real-world data
		- oversampling the minority
		- under sampling the majority
		- applying cost-sensitive learning
### Advantages

- scalability
	- efficiently processes large datasets by updating model parameters incrementally rather than relying on full batch computations
	- well suited for real-time applications
- speed and memory efficiency
	- particularly when dealing with high-dimensional data
- supports online learning
	- model can be continuously updated with new data
### Disadvantages

- high variance and instability
	- updating parameters with single samples at a time introduces stochastic noise
	- leads to fluctuating loss values
	- potentially converge to a suboptimal solution
- heavily depends on hyperparameter selection
	- particularly the learning rate and the number of iterations
	- poorly chosen learning rate may result in slow convergence or even divergence of the model
- not optimal for small datasets
	- relies on random updates
	- may lead to poor generalization
	- needs batch or mini-batch gradient descent methods to stable and reliable result

# Rule-Based Classification

- technique in machine learning and data analysis where classification decisions are made based on a set of predefined rules
	- typically derived from the training data and follow an "if-then" structure
	- allow interpretable decision-making
- uses explicit logical conditions to categorize data points
	- rules often extracted from decision trees, association rule mining (Apriori algorithm) or expert knowledge

```python
import pandas as pd

# Sample dataset with customer transactions
data = {
    "Customer_ID": [101, 102, 103, 104, 105],
    "Total_Purchases": [15, 2, 8, 12, 1],
    "Total_Spend": [1200, 150, 800, 950, 50],
    "Last_Purchase_Days_Ago": [30, 210, 90, 45, 365]
}
df = pd.DataFrame(data)

# Define classification function
def classify_customer(row):
    if row["Total_Purchases"] > 10 and row["Total_Spend"] > 1000:
        return "Loyal Customer"
    elif row["Total_Purchases"] < 3 and row["Last_Purchase_Days_Ago"] > 180:
        return "At-Risk Customer"
    elif row["Total_Purchases"] >= 5 and row["Total_Spend"] > 500:
        return "Regular Customer"
    else:
        return "Occasional Customer"

# Apply classification
df["Customer_Category"] = df.apply(classify_customer, axis=1)

# Display the classified results
import ace_tools as tools
tools.display_dataframe_to_user(name="Rule-Based Classification Results", dataframe=df)
```
### Output

- output classifies customer based on predefined rules
### Advantage

- interpretability
	- classification is based on explicitly defined rules &rarr; reasoning behind each decision is clear and easy to explain
	- highly valuable in regulatory environments where decision transparency is critical
- computationally efficient
	- when applied to structured datasets
	- does not require iterative training processes
### Disadvantages

- rigidity
	- rules must be manually defined
	- making system inflexible when dealing with dynamic or evolving data patterns
	- may require frequent updates
- struggles with handling complex relationships between variables
	- especially when interactions are non-linear or involve high-dimensional data