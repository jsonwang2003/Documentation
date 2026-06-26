> [!INFO]
> Measures the area under the precision-recall curve, useful for evaluating models on imbalanced datasets.

## How It Works

The Precision-Recall Area Under Curve (PR-AUC) summarizes the **trade-off between precision and recall across different classification thresholds**.

- **[[Precision]]**: How many predicted positives are correct  
- **[[Recall]]**: How many actual positives are captured  

The curve plots **precision vs. recall**, and the area under it reflects overall performance.

## What to Look For

- Higher PR-AUC = better performance, especially on imbalanced data  
- More informative than ROC-AUC when positive class is rare  
- Ideal for tasks like fraud detection, medical diagnosis, or anomaly detection

## Application Models

- [[Random Forest Classification]] 
- [[Support Vector Machine (SVM)]]
- [[Neural Network Regression]]