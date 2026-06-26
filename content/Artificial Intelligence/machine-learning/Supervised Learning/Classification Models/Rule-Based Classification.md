> [!INFO]
> Technique in machine learning and data analysis where **classification decisions are made based on predefined rules**  
> Rules typically follow an **"if-then" structure** and are derived from training data, expert knowledge, or rule-mining algorithms

- **Developed by**: Rooted in **expert systems** and **symbolic AI** from the 1970s–1980s
- **Core Principle**: Uses **explicit logical conditions** to categorize data points
- **Search Strategy**:
	- Rules can be extracted from:
		- **Decision trees**
		- **Association rule mining** (e.g., **Apriori algorithm**)
		- **Domain expert knowledge**
	- Enables **interpretable decision-making** with transparent logic

## Workflow

1. **Rule Definition**
	- Manually or algorithmically define classification rules
	- Use logical conditions based on feature thresholds
2. **Rule Application**
	- Apply rules to each data point
	- Assign class labels based on matched conditions

## Code Example

```python
import pandas as pd

# Sample dataset with customer transactions
data = {
    "Customer_ID": [101, 102, 103, 104, 105],
    "Total_Purchases": [15, 2, 8, 12, 1],
    "Total_Spend": [1200, 150, 800, 950, 50],
    "Last_Purchase_Days_Ago": [30, 210, 90, 45, 365]
}
df = pd.DataFrame(data)

# Define classification function
def classify_customer(row):
    if row["Total_Purchases"] > 10 and row["Total_Spend"] > 1000:
        return "Loyal Customer"
    elif row["Total_Purchases"] < 3 and row["Last_Purchase_Days_Ago"] > 180:
        return "At-Risk Customer"
    elif row["Total_Purchases"] >= 5 and row["Total_Spend"] > 500:
        return "Regular Customer"
    else:
        return "Occasional Customer"

# Apply classification
df["Customer_Category"] = df.apply(classify_customer, axis=1)

# Display the classified results
import ace_tools as tools
tools.display_dataframe_to_user(name="Rule-Based Classification Results", dataframe=df)
```
## Advantages

- **Interpretability**
	- Decisions are based on **explicitly defined rules**
	- Easy to explain and audit
	- Valuable in **regulatory environments**
- **Computationally efficient**
	- No iterative training required
	- Fast execution on **structured datasets**

## Disadvantages

- **Rigid and inflexible**
	- Rules must be manually defined
	- Requires frequent updates for **dynamic data**
- Limited in handling **complex relationships**
	- Struggles with **non-linear interactions** or **high-dimensional data**