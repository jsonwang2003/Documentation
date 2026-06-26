> [!INFO]
> A foundational technique in association rule learning used to identify frequent itemsets and derive rules from large transactional datasets.

- **Developed by**: Agrawal and Srikant (1994)
- **Core Principle**: A subset of a frequent itemset must also be frequent
- **Search Strategy**: Level-wise, breadth-first expansion

## Workflow

1. Frequent Itemset Generation
	- Expand itemset one item at a time
	- Prune candidates that do not meet the **minimum support threshold**
2. Rule Derivation
	- Generate rules from frequent itemset
	- Evaluate using metrics:
		- **Support**: Frequency of itemset in transactions
		- **Confidence**: Likelihood that consequent appears when antecedent does
		- **Lift**: Measures strength of association vs. random chance

## Code Example

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction dataset
data = {
    'Transaction': [1, 2, 3, 4, 5],
    'Milk': [1, 1, 0, 1, 1],
    'Bread': [1, 0, 1, 1, 1],
    'Butter': [1, 0, 1, 0, 1],
    'Jam': [0, 1, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data).set_index('Transaction')

# Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Association Rules", dataframe=rules)
```

## Advantages

- Produces interpretable rules for actionable insights
- Applicable across domains (retail, healthcare, web usage)
- Metrics (support, confidence, lift) help prioritize relationships
## Disadvantages

- High computational cost for large datasets
- Poor scalability in high-dimensional spaces
- Requires manual tuning of support/confidence threshold