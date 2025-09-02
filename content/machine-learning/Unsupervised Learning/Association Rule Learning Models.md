# Apriori Algorithm

> [!INFO]
> Fundamental association rule learning technique used in data mining to identify frequent itemsets and derive assocation rules from large transactional datasets

- Developed by Agrawal and Srikant (1994)
- Algorithm Process: **level-wise search**
	- Operates on the principle that a subset of a frequent itemset must also be frequent
	1. Expands frequent itemset one item at a time
	2. Prunes infrequent ones based on the minimum support threshold
		- **Support**: measures the frequency of an itemset appearing in transactions
		- **Confidence**: assesses how often the rule is correct when the antecedent occurs
		- **Lift**: quantifies the dependency between itemset

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction dataset (Each row represents a customer's purchases)
data = {
    'Transaction': [1, 2, 3, 4, 5],
    'Milk': [1, 1, 0, 1, 1],
    'Bread': [1, 0, 1, 1, 1],
    'Butter': [1, 0, 1, 0, 1],
    'Jam': [0, 1, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data).set_index('Transaction')

# Apply Apriori Algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Generate association rules based on confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Association Rules", dataframe=rules)
```
### Advantages

- Provides interpretable rules
	- Easy for deriving insights
- Can be applied to various domains
- Metrics (support, confidence, and lift) allow quantify and prioritize important relationships among items
### Disadvantages

- Computational complexity
- Cannot deal with high-dimensional datasets
- Dependence on manually set thresholds

# Equivalence Class Clustering and Bottom-Up Lattice Traversal (ECLAT)

> [!INFO]
> Highly efficient algorithm used for frequent itemset mining in transactional databases though [[Depth First Search]]

- Algorithm Process
	- Use a vertical data format representation
		- Each item is associated with a list of transaction identifiers (TIDs) instead of a horizontal layout
		- Allows efficiently intersect these TID lists to determine frequent itemset
	- Bottom-up manner
		- Systematically combines smaller frequent itemset to generate larger ones
		- Eliminated need for candidate generation

```python
from collections import defaultdict
from itertools import combinations

class ECLAT:
    def __init__(self, min_support=2):
        self.min_support = min_support
        self.itemsets = defaultdict(set)

    def fit(self, transactions):
        tid = 0
        for transaction in transactions:
            for item in transaction:
                self.itemsets[item].add(tid)
            tid += 1

        self.frequent_itemsets = {}
        self._eclat_recursive({}, list(self.itemsets.items()))

    def _eclat_recursive(self, prefix, items):
        for i, (item, tids) in enumerate(items):
            new_prefix = prefix | {item}
            support = len(tids)
            if support >= self.min_support:
                self.frequent_itemsets[frozenset(new_prefix)] = support
                new_items = [
                    (other_item, tids & other_tids)
                    for other_item, other_tids in items[i+1:]
                    if len(tids & other_tids) >= self.min_support
                ]
                self._eclat_recursive(new_prefix, new_items)

    def get_frequent_itemsets(self):
        return self.frequent_itemsets

# Sample transactions
transactions = [
    {'bread', 'milk', 'butter'},
    {'bread', 'diaper', 'beer'},
    {'milk', 'butter'},
    {'bread', 'milk', 'diaper', 'butter'},
    {'bread', 'milk', 'diaper'}
]

eclat = ECLAT(min_support=2)
eclat.fit(transactions)
frequent_itemsets = eclat.get_frequent_itemsets()

# Display results
import pandas as pd
df_results = pd.DataFrame(frequent_itemsets.items(), columns=['Itemset', 'Support'])
import ace_tools as tools
tools.display_dataframe_to_user(name="ECLAT Frequent Itemsets", dataframe=df_results)
```    
### Advantages

- Efficient memory utilization
- Able to handle large datasets effectively
- Simple to implement
### Disadvantages

- Rely on intersecting transaction ID lists
- May struggle with dense datasets where many items frequently co-occur
- Lack of direct association rule generation
	- Further processing required to extract actionable rules

# Frequent Pattern Growth

>[!INFO]
>- Used for mining frequent itemset in transactional datasets
>- Alternative to the Apriori algorithm and addresses its inefficiencies by eliminating the need for candidate generation
- Utilizes a data structure: [[Frequent Pattern Tree(FP-Tree)]]
	- Compresses the dataset by aggregating common prefixes of transactions
	- Significantly reducing the search space
- Algorithm Process
	1. Constructs the FP-tree by scanning the dataset twice to count item frequencies
	2. Build a hierarchical tree structure
	3. Recursively mins the tree for frequent itemset by identifying conditional patterns bases and generating frequent patterns

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
### Advantages

- More efficient than Apriori
	- Eliminates the costly process of candidate generation
	- Reduces computational overhead using the FP-tree
- Well suited for large datasets
	- Only requires 2 scans of the database and compresses data effectively
- Can handle datasets with long frequent itemset
### Disadvantages

- Can be memory-intensive
	- Constructing and traversing the FP-tree requires additional data structures
- Less interpretable compared to Apriori
- Performance and degrade when dealing with datasets that contain a large number of unique items
	- More complex tree structure