# Classification

## Accuracy

> [!INFO]
> Measures the proportion of correct predictions out of total predictions.

### Application Models

- CNNs  
- DNNs  
- Transformers

## Precision

> [!INFO]
> Measures the proportion of correctly predicted positive observations to the total predicted positives.

### Application Models

- CNNs  
- DNNs  
- Transformers

## Recall

> [!INFO]
> Measures the proportion of correctly predicted positive observations to all actual positives.

### Application Models

- CNNs  
- DNNs  
- Transformers

## F1-Score

> [!INFO]
> Harmonic mean of precision and recall, providing a balanced measure.

### Application Models

- CNNs  
- DNNs  
- Transformers

## ROC-AUC

> [!INFO]
> Represents the area under the receiver operating characteristic (ROC) curve, assessing classification performance.

### Application Models

- CNNs  
- DNNs  
- Transformers

## PR-AUC (Precision-Recall Area Under Curve)

> [!INFO]
> Measures the area under the precision-recall curve, useful for imbalanced datasets.

### Application Models

- Random Forest  
- SVM  
- Neural Networks

## Cohen’s Kappa

> [!INFO]
> Measures agreement between predicted and actual classifications while accounting for chance agreement.

### Application Models

- Random Forest  
- SVM  
- Neural Networks

## Matthews Correlation Coefficient (MCC)

> [!INFO]
> Evaluates classification models on imbalanced datasets by considering all confusion matrix elements.

### Application Models

- Logistic Regression  
- Random Forest  
- SVM

## Confusion Matrix

> [!INFO]
> A table that shows TP, FP, TN, and FN values to assess classification performance.

### Application Models

- Decision Trees  
- Neural Networks  
- SVM  
- Logistic Regression

## Brier Score

> [!INFO]
> Measures the accuracy of probabilistic predictions in classification tasks. Lower values indicate better calibration.

### Application Models

- Logistic Regression  
- Random Forest  
- Gradient Boosting

# Regression

## Mean Absolute Error (MAE)

> [!INFO]
> Measures the average absolute differences between predicted and actual values in regression tasks.

### Application Models

- DNNs  
- RNNs

## Mean Squared Error (MSE)

> [!INFO]
> Measures the average squared difference between actual and predicted values, used for regression models.

### Application Models

- DNNs  
- RNNs

## Root Mean Squared Error (RMSE)

> [!INFO]
> Square root of MSE, making it interpretable in the same unit as the target variable. Penalizes large errors.

### Application Models

- Linear Regression  
- Decision Trees  
- Random Forest  
- Neural Networks

## R-Squared (R²)

> [!INFO]
> Represents the proportion of variance explained by a regression model, assessing its fit.

### Application Models

- DNNs  
- RNNs

## Adjusted R-Squared

> [!INFO]
> Modified R² that adjusts for the number of predictors in the model to prevent overfitting.

### Application Models

- Multiple Linear Regression  
- Decision Trees  
- Random Forest

## Mean Absolute Percentage Error (MAPE)

> [!INFO]
> Measures the percentage error between actual and predicted values, useful for interpretability.

### Application Models

- Linear Regression  
- Time Series Models

# Clustering

## Silhouette Score

> [!INFO]
> Measures how similar a data point is to its own cluster compared to other clusters. A higher score indicates better-defined clusters.

### Application Models

- K-Means  
- Hierarchical Clustering  
- DBSCAN  
- Gaussian Mixture Models (GMM)

## Davies-Bouldin Index

> [!INFO]
> Calculates the ratio between intra-cluster and inter-cluster distances. A lower index value indicates better clustering.

### Application Models

- K-Means  
- Hierarchical Clustering  
- DBSCAN  
- Gaussian Mixture Models (GMM)

## Calinski-Harabasz Index

> [!INFO]
> Measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion. A higher index value indicates better clustering.

### Application Models

- K-Means  
- Hierarchical Clustering  
- DBSCAN  
- Gaussian Mixture Models (GMM)

## Dunn Index

> [!INFO]
> Measures the minimum inter-cluster distance relative to the maximum intra-cluster distance. A higher index value suggests better separation.

### Application Models

- K-Means  
- Hierarchical Clustering  
- DBSCAN  
- Gaussian Mixture Models (GMM)

## Elbow Method

> [!INFO]
> Determines the optimal number of clusters by plotting within-cluster variance as a function of the number of clusters.

### Application Models

- K-Means

## Gap Statistic

> [!INFO]
> Compares the performance of different clustering results using the expected dispersion under a reference null distribution.

### Application Models

- K-Means  
- Hierarchical Clustering

# Physics-Informed & Scientific

## Relative Error

> [!INFO]
> Measures the deviation between predicted and actual physical quantities relative to actual values.

### Application Models

- PINNs  
- Scientific ML models

## Energy Conservation Error

> [!INFO]
> Assesses how well a model conserves energy in simulations, ensuring physical consistency.

### Application Models

- PINNs  
- Simulation-driven models

## Residual Loss

> [!INFO]
> Quantifies the error in governing equations, used in physics-informed neural networks (PINNs).

### Application Models

- PINNs  
- PDE solvers

## Physics-Informed Loss

> [!INFO]
> Incorporates physical constraints into loss functions to ensure model adherence to known laws.

### Application Models

- PINNs

## Structural Similarity Index (SSIM)

> [!INFO]
> Evaluates the perceptual similarity of generated physical field data, used in scientific computing.

### Application Models

- Physics-based neural networks  
- Image-based simulation models

# NLP & Language Modeling

## Perplexity

> [!INFO]
> Measures how well a language model predicts a sample; lower values indicate better predictions.

### Application Models

- Transformers  
- RNNs  
- LSTMs

## BLEU Score

> [!INFO]
> Evaluates the quality of machine translation by comparing output with reference translations.

### Application Models

- Transformers  
- RNNs

# Image & Dimensionality Reduction

## PSNR (Peak Signal-to-Noise Ratio)

> [!INFO]
> Measures the quality of reconstructed images compared to the original, used in image processing.

### Application Models

- Autoencoders  
- GANs  
- CNNs

## Reconstruction Error

> [!INFO]
> Measures the difference between original data and reconstructed data, often used in dimensionality reduction methods like PCA and autoencoders.

### Application Models

- PCA  
- Autoencoders

# Association Rule Mining

## Lift Metric

> [!INFO]
> Assesses how much more likely a rule-based association is to occur compared to a random occurrence.

### Application Models

- Apriori  
- FP-Growth

## Support & Confidence

> [!INFO]
> Evaluates the strength of association rules based on the proportion of transactions that contain both antecedent and consequent items.

### Application Models

- Apriori  
- Eclat