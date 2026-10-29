> [!INFO]
> A powerful technique in frequent itemset mining that uses vertical data format and depth-first traversal to efficiently discover frequent patterns.

- **Developed by**: Zaki et al. (1997)
- **Core Principle**: Frequent itemsets are discovered by intersecting transaction ID sets
- **Search Strategy**: Depth-first, bottom-up lattice traversal
## Workflow

1. Vertical Data Transformation  
   - Convert transactions into item &rarr; TID (Transaction ID) mappings  
   - Enables fast computation via set intersections
1. Frequent Itemset Generation  
   - Recursively combine items with overlapping TIDs  
   - Prune combinations that do not meet the **minimum support threshold**
1. Post-Processing (Optional)  
   - Association rules can be derived from frequent itemsets  
   - Requires additional metrics like:  
     - **Support**: Frequency of itemset in transactions  
     - **Confidence**: Likelihood that consequent appears when antecedent does  
     - **Lift**: Measures strength of association vs. random chance
## Code Example

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
## Advantages

- Efficient memory usage due to vertical format
- Scales well with large datasets
- Simple and elegant implementation

## Disadvantages

- Performance depends on intersecting TID lists
- May struggle with dense datasets where many items co-occur
- Does not directly generate association rules (requires post-processing)