> [!INFO]
> Represents the area under the receiver operating characteristic (ROC) curve, assessing classification performance.

## How It Works

- Plots the True Positive Rate (TPR) against the False Positive Rate (FPR) across thresholds  
- $TPR = \frac{\text{True Positive}}{(\text{True Positive} + \text{False Negative})}$
- $TPR = \frac{\text{False Positive}}{(\text{False Positive} + \text{True Negative})}$
- AUC quantifies the model’s ability to distinguish between classes

## What to Look For

- AUC close to 1 = excellent performance  
- Threshold-independent  
- Best for binary classification tasks

## Application Models

- [[Convolutional Neural Network (CNN)]]  
- [[Deep Neural Network (DNN)]]  
- [[Transformer]]