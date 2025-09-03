# Classification

- fundamental supervised learning technique used to **predict categorical labels** by **mapping input features to predefined classes**
- widely applied in tasks such as
	- spam detection
	- medical diagnosis
	- sentiment analysis
- can be
	- **binary**: predict whether an email is spam or not
	- **multiclass**: classifying handwritten digits into ten categories
	- **multi-label**: identifying multiple objects in an image
- most common algorithms for classification
	- [[Classification Models#Logistic Regression|Logistic Regression]]
	- [[Classification Models#Classification Decision Trees|Classification Decision Tree]]
	- [[Classification Models#Random Forest Classification|Random Forest]]
	- [[Classification Models#Support Vector Machines (SVM)|Support Vector Machine (SVM)]]
	- [[Classification Models#Naïve Bayes|Naïve Bayes]]
	- [[Classification Models#k-Nearest Neighbors (k-NN)|k-Nearest Neighbors (k-NN)]]
- Deep Learning models leveraged
	- [[machine-learning/Deep Learning/Supervised Deep Learning Models/index#Convolutional Neural Network (CNN)|Convolutional Neural Networks (CNNs)]] for image classification
	- Transformer-based architectures for **natural language processing**
- Classification models evaluated using metrics
	- [[Model Evaluation#Accuracy|Accuracy]]
	- [[Model Evaluation#Precision|Precision]]
	- [[Model Evaluation#Recall|Recall]]
	- [[Model Evaluation#F1-Score|F1-Score]]
	- [[Model Evaluation#ROC-AUC|ROC-AUC]]
# Regression

- used to predict **continuous numerical values based on input features**
- essential for forecasting and estimating quantities in various domains
- use various regression models for different problems at hand
	- from [[Regression Models#Linear Regression|Linear Regression]] to [[machine-learning/Deep Learning/Supervised Deep Learning Models/index#Neural Network Regression|Neural Network Regression]]
- Performance metrics which measures how well the model fits the data

	- [[Model Evaluation#Mean Absolute Error (MAE)|Mean Absolute Error (MAE)]]
	- [[Regression Model Evaluation#Mean Squared Error ($MSE$)|Mean Squared Error (MSE)]]
	- [[Model Evaluation#Root Mean Squared Error (RMSE)|Root Mean Squared Error]]
	- [[Regression Model Evaluation#Coefficient of Determination ($R 2$)|Coefficient of Determination]]
# Ordinal Regression

- supervised learning approach that **combines elements of both classification and regression by predicting ordered categorical outcomes**
- takes into account the **meaningful ranking of categories**
	- have inherent order but no precise numerical difference between them
		- traditional classification models are not well suited for such tasks
		- specialized models are used
			- Ordinal Logistic Regression
			- [[Classification Models#Gradient Boosting Machines (GBM)|Gradient Boosting Methods]]
			- Neural Networks with ranking loss functions
- model evaluated using metrics to measure their effectiveness in **preserving the ordinal structure in predictions**
	- Spearman's rank correlation
	- mean squared error of ranks
# Time Series Forecasting

- type of supervised learning where models **predict future values based on historical patterns**
- crucial in fields like
	- weather prediction
	- stock market analysis
	- demand forecasting
	- economic trend analysis
- incorporates **temporal dependencies**
	- past observations influence future predictions
- Statistical models
	- reliable baseline for forecasting
		- Autoregressive Integrated Moving Average (ARIMA)
		- Exponential Smoothing
	- improved performance for capturing long-term dependencies in sequential data
		- [[machine-learning/Deep Learning/Supervised Deep Learning Models/index#Long Short-Term Memory (LSTM)|Long Short-Term Memory Networks (LSTMs)]]
		- Gated Recurrent Units (GRUs)
	- for handling irregular trends and seasonality
		- Facebook's Prophet
		- XGBoost
- Performance evaluation metrics
	- [[Model Evaluation#Mean Absolute Percentage Error (MAPE)|Mean Absolute Percentage Error (MAPE)]]
	- [[Model Evaluation#Mean Squared Error (MSE)|Mean Squared Error (MSE)]]
	- [[Model Evaluation#Root Mean Squared Error (RMSE)|Root Mean Squared Error (RMSE)]]

# Survival Analysis

- specialized branch of supervised learning used to **predict time-to-event data**
	- estimating time until patient's recovery
	- equipment failure
	- customer churn
- accounts for **censored data, where the event of interest may not have occurred for all observations during the study period**
- Methods
	- **Kaplan-Meier estimator**[^1]
	- **Cox Proportional Hazards Model**[^2]
	- [[Classification Models#Random Forest Classification|Random Forest Classification]] and [[Deep Learning/index|Deep Learning-based survival models]]
		- allow for higher accuracy in complex, high-dimensional datasets
- Evaluation Metrics
	- Concordance index (C-index)
	- Log-Rank Test

# Anomaly Detection

- identifying **rare or unexpected patterns** based on labeled data
- widely used in fraud detection, cybersecurity threat identification, and predictive maintenance
- requires **labeled examples of both normal and anomalous instances to train the model**
- Techniques
	- One-Class [[Classification Models#Support Vector Machines (SVM)|Support Vector Machine (SVM)]]
	- [[Isolation Forest]] (if trained with labeled data)
	- ensemble methods
		- combines multiple models for improved neural networks
- Model Performance evaluated using
	- [[Model Evaluation#PR-AUC (Precision-Recall Area Under Curve)|Precision-Recall Area Under Curve]]
	- [[Model Evaluation#F1-Score|F1-Score]]
	- [[Model Evaluation#ROC-AUC|ROC-AUC]]
	- ensure effective identification of anomalies with minimal false positives

[^1]: a **non-parametric method** commonly used for _estimating survival functions_

[^2]: Cox Proportional Hazards Model: provides a **semi-parametric approach** for analyzing the _effect of **covariates on event occurrence**_
