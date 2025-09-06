---
title: Classification Models
---
> [!INFO]
> Classification models are supervised learning algorithms that **predict categorical labels by learning decision boundaries from labeled data**. They are foundational in tasks like spam detection, medical diagnosis, and sentiment analysis.

## Overview

Classification focuses on assigning input data to one or more predefined categories. Models learn patterns from labeled examples and generalize to unseen data. Tasks may be:
- **Binary** (e.g., spam vs. not spam)
- **Multiclass** (e.g., digit recognition)
- **Multi-label** (e.g., tagging multiple objects in an image)

Common applications include **fraud detection**, **document classification**, **image recognition**, and **disease diagnosis**.

---

## Included Algorithms

- ### [[machine-learning/Supervised Learning/Classification Models/Logistic Regression|Logistic Regression]]

> Models the probability of class membership using a sigmoid function. Simple, interpretable, and effective for linearly separable data.

- ### [[machine-learning/Supervised Learning/Classification Models/Classification Decision Tree|Classification Decision Tree]]

> Splits data based on feature thresholds to form a tree of decisions. Easy to interpret but prone to overfitting.

- ### [[Random Forest Classification|Random Forest Classification]]

> Ensemble of decision trees trained on bootstrapped samples. Reduces overfitting and improves generalization.

- ### [[machine-learning/Supervised Learning/Classification Models/Support Vector Machine (SVM)|Support Vector Machine (SVM)]]

> Finds optimal hyperplanes to separate classes. Effective in high-dimensional spaces and with clear margins.

- ### [[machine-learning/Supervised Learning/Classification Models/Naïve Bayes|Naïve Bayes]]

> Probabilistic model based on Bayes’ theorem with strong independence assumptions. Fast and effective for text classification.

- ### [[machine-learning/Supervised Learning/Classification Models/k-Nearest Neighbors (k-NN)|k-Nearest Neighbors (k-NN)]]

> Classifies based on the majority label among nearest neighbors. Non-parametric and intuitive but computationally expensive at inference.

- ### [[machine-learning/Supervised Learning/Classification Models/Gradient Boosting Machines (GBM)|Gradient Boosting Machines (GBM)]]

> Builds models sequentially to correct errors of previous ones. Powerful and flexible but sensitive to hyperparameters.

- ### [[Adaptive Boosting (AdaBoost)|Adaptive Boosting (AdaBoost)]]

> Combines weak learners by focusing on misclassified instances. Works well with simple base models but can be sensitive to noise.

- ### [[Stochastic Gradient Descent (SGD) Classifier|Stochastic Gradient Descent (SGD) Classifier]]

> Optimizes linear classifiers using incremental updates. Scales well to large datasets but requires careful tuning.

- ### [[machine-learning/Supervised Learning/Classification Models/Rule-Based Classification|Rule-Based Classification]]

> Uses human-readable rules to assign labels. Transparent and interpretable, often used in expert systems.

--- 

## Key Concepts

- **Decision Boundaries**: Separators between classes in feature space
- **Loss Functions**: Guide model optimization (e.g., cross-entropy)
- **Hyperplanes**: Used in SVM to separate classes
- **Ensemble Learning**: Combines multiple models for better performance
- **Bias-Variance Tradeoff**: Balancing underfitting and overfitting

---

## Applications

- Email spam filtering
- Disease diagnosis from medical records
- Sentiment analysis in social media
- Image classification in computer vision
- Fraud detection in financial systems

---

## Suggested Links

- [[machine-learning/Supervised Learning/index|Supervised Learning]] — Broader context for learning from labeled data
- [[machine-learning/Model Evaluation|Model Evaluation]] — Metrics like accuracy, precision, recall, F1-score, ROC-AUC
- [[machine-learning/Feature Engineering/index|Feature Engineering]] — Crucial for improving classification performance