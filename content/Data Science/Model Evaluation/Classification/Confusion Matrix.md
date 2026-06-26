> [!INFO]
> A table that shows **True Positive, False Positive, True Negative, and False Negative values** to assess classification performance.

## How It Works

The confusion matrix is a 2×2 table for binary classification:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | TP                | FN                |
| Actual Negative | FP                | TN                |

- **TP**: Correctly predicted positive  
- **FP**: Incorrectly predicted positive  
- **TN**: Correctly predicted negative  
- **FN**: Incorrectly predicted negative  

## What to Look For

- Foundation for **most classification metrics**  
- Reveals types of errors (false positives vs. false negatives)  
- Use to derive [[Precision]], [[Recall]], [[F1-Score]], [[Matthews Correlation Coefficient (MCC)]], etc.

## Application Models

- [[Classification Decision Tree]]
- [[Neural Network Regression]]
- [[Support Vector Machine (SVM)]]
- [[Logistic Regression]]