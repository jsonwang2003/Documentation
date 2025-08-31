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
	- Logistic Regression
	- Decision Tree
	- Random Forest
	- Support Vector Machines
	- Naïve Bayes
	- k-Nearest Neighbors (k-NN)
- Deep Learning models leveraged
	- Convolutional Neural Networks (CNNs) for image classification
	- Transformer-based architectures for natural language processing
- Classification models evaluated using metrics
	- accuracy
	- precision
	- recall
	- F1-score
	- ROC-AUC
# Regression

- used to predict **continuous numerical values based on input features**
- essential for forecasting and estimating quantities in various domains
- use various regression models for different problems at hand
	- from [[Linear Regression/index|Linear Regression]] to [[Neural Network Regression/index|Neural Network Regression]]
- Performance metrics which measures how well the model fits the data

	- Mean Absolute Error (MAE)
	- Mean Squared Error (MSE)
	- Root Mean Squared Error (RMSE)
	- R-squared ($R^2$)
# Ordinal Regression

- supervised learning approach that **combines elements of both classification and regression by predicting ordered categorical outcomes**
- takes into account the **meaningful ranking of categories**
	- have inherent order but no precise numerical difference between them
		- traditional classification models are not well suited for such tasks
		- specialized models are used
			- Ordinal Logistic Regression
			- Gradient Boosting Methods
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
		- Long Short-Term Memory Networks (LSTMs)
		- Gated Recurrent Units (GRUs)
	- for handling irregular trends and seasonality
		- Facebook's Prophet
		- XGBoost
- Performance evaluation metrics
	- [[Supervised Model Evaluation|Mean Absolute Percentage Error (MAPE)]]
	- [[Supervised Model Evaluation|Mean Squared Error (MSE)]]
	- [[Supervised Model Evaluation|Root Mean Squared Error (RMSE)]]

# Survival Analysis

- specialized branch of supervised learning used to **predict time-to-event data**
	- estimating time until patient's recovery
	- equipment failure
	- customer churn
- accounts for **censored data, where the event of interest may not have occurred for all observations during the study period**
- Methods
	- Kaplan-Meier estimator
		- non-parametric
		- used for estimating survival functions
	- Cox Proportional Hazards Model
		- semi-parametric approach for analyzing effects of covariates on event occurrence
	- [[Random Survival Forests]] and [[Deep Learning/index|Deep Learning-based survival models]]
		- allow for higher accuracy in complex, high-dimensional datasets
- Evaluation Metrics
	- Concordance index (C-index)
	- Log-Rank Test

# Anomaly detection

- identifying **rare or unexpected patterns** based on labeled data
- widely used in fraud detection, cybersecurity threat identification, and predictive maintenance
- requires **labeled examples of both normal and anomalous instances to train the model**
- Techniques
	- One-Class [[Support Vector Machines (SVM)]]
	- [[Isolation Forest]] (if trained with labeled data)
	- ensemble methods
		- combines multiple models for improved neural networks
- Model Performance evaluated using
	- precision-recall curves
	- F1-score
	- area under the ROC curve (AUC-ROC)
	- ensure effective identification of anomalies with minimal false positives