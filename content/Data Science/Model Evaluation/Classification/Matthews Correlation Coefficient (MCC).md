> [!INFO]
> Evaluates classification models on imbalanced datasets by considering all confusion matrix elements.

## How It Works

Matthews Correlation Coefficient (MCC):

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

- **TP**: True Positives  
- **TN**: True Negatives  
- **FP**: False Positives  
- **FN**: False Negatives  

## What to Look For

- Range: -1 (total disagreement) to +1 (perfect prediction)  
- Robust metric for imbalanced datasets  
- Preferred when accuracy is misleading

## Application Models

- [[Logistic Regression]]
- [[Random Forest Classification]]
- [[Support Vector Machine (SVM)]]