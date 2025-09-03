> [!INFO]
> Used for mining frequent itemsets in transactional datasets.  
> An alternative to the Apriori algorithm that eliminates candidate generation.  
> Utilizes a data structure called the **Frequent Pattern Tree (FP-Tree)** to compress transactions and reduce the search space.

- **Developed by**: Han et al. (2000)  
- **Core Principle**: Common prefixes in transactions are aggregated to build a compact tree  
- **Search Strategy**: Recursive mining of conditional pattern bases

## Workflow

1. [[Frequent Pattern Tree]] Construction  
   - Scan the dataset twice to count item frequencies  
   - Build a hierarchical tree structure by inserting transactions in frequency order
1. Frequent Pattern Mining  
   - Traverse the tree recursively  
   - Identify conditional pattern bases  
   - Generate frequent itemset without candidate generation
1. Rule Derivation (Optional)  
   - Generate association rules from frequent itemset  
   - Evaluate using metrics:  
     - **Support**: Frequency of itemset in transactions  
     - **Confidence**: Likelihood that consequent appears when antecedent does  
     - **Lift**: Measures strength of association vs. random chance
## Code Example

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Sample transactional dataset
dataset = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'bread', 'butter', 'jam'],
    ['bread', 'jam'],
    ['milk', 'bread', 'jam']
]

# Convert to DataFrame format
items = sorted(set(item for transaction in dataset for item in transaction))
df = pd.DataFrame([{item: (item in transaction) for item in items} for transaction in dataset])

# Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Frequent Itemsets", dataframe=frequent_itemsets)
tools.display_dataframe_to_user(name="Association Rules", dataframe=rules)
```
## Advantages

- More efficient than Apriori
    - Eliminates costly candidate generation
    - Reduces computational overhead using FP-tree
- Well suited for large datasets
    - Requires only two scans of the database
    - Compresses data effectively
- Can handle datasets with long frequent itemset
## Disadvantages

- Can be memory-intensive
    - FP-tree construction and traversal require additional data structures
- Less interpretable compared to Apriori
- Performance may degrade with datasets containing many unique items
    - Results in a more complex tree structure