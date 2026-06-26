# Correlation Analysis

> [!INFO]
> statistical technique used to measure the strength and direction of the relationship between 2 or more variables

- widely used in business and data science to determine patterns and dependencies that can inform strategic decision-making
- range from -1 to +1
	- values closer to +1 indicate strong positive relationship
	- values closer to -1 indicate strong negative relationship
	- values around 0 suggest no significant relationship
- commonly used correlation metrics
	- Pearson's correlation (measures linear relationships)
	- Spearman's rank correlation (monotonic relationships)
	- Kendall's Tau correlation (ordinal data)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Generate synthetic data
np.random.seed(42)
advertising_spend = np.random.randint(5000, 20000, 50)  # Advertising budget
revenue = advertising_spend * 3.5 + np.random.normal(0, 5000, 50)  # Revenue with some noise

# Create a DataFrame
data = pd.DataFrame({'Advertising Spend': advertising_spend, 'Revenue': revenue})

# Calculate Pearson Correlation
corr_coefficient, p_value = pearsonr(data['Advertising Spend'], data['Revenue'])

# Display correlation matrix
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Print results
print(f"Pearson Correlation Coefficient: {corr_coefficient:.2f}")
print(f"P-Value: {p_value:.5f}")
```

- The Pearson correlation coefficient **provides insight into the strength of the relationship** between advertising spend and revenue
	- If output shows a correlation coefficient close to
		- +1: indicating a strong positive &rarr; higher advertising expenditures are generally associated with higher revenue
- p-value tests the statistical significance of the correlation
	- p-value < 0.05: statistically significant

> [!IMPORTANT]
> correlation DOES NOT establish a causal link, other factors could also influence revenue

- strengths of correlation analysis
	- ability to **quickly identify relationships between variables**
		- valuable tool for **exploratory analysis**
	- provides quantitative insights
- Weakness of correlation analysis
	- inability to determine causation
		- high correlation between 2 variables $\neq$ one causes the other
		- hidden confounding variables could be influencing both
	- most effective for linear relationships
		- if the relationship is non-linear, standard correlation coefficients may provide misleading results
	- impact of outliers
		- can distort correlation values and lead to incorrect conclusions
	- ***To mitigate these issues***, correlation analysis should be complemented with **regression modeling** and **experimental design** to establish causal relationships