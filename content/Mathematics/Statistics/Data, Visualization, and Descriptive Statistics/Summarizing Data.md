> [!INFO] Definition
> Data Summarization is the process of **condensing large amounts of data** into *smaller*, more *informative representations* that capture the essential information of the dataset. This is crucial for making the data more **understandable**, **interpretable**, and **manageable**

---
## Methods of Data Summarization
### [[Data Visualization|Shape]]
Describes the **distribution of data**—how the data points are spread out across the range of values.
- **Graphical Representation**: Tools like histograms, box plots, and density plots are used to visualize the "silhouette" of the data.
- **Skewness**: Measures the lack of symmetry.
    - **Right-skewed** (Positive): The tail extends toward higher values.
    - **Left-skewed** (Negative): The tail extends toward lower values.
- **Kurtosis**: Measures the "tailedness" or "peakedness" of the distribution.
    - **Leptokurtic**: Sharp peak with heavy tails.
    - **Platykurtic**: Flat peak with thin tails.
### [[Central Tendency]]
Measures that provide a single value representing the **center or "typical" value** of a dataset.
- **Mean**: The arithmetic average (Sum of values / Count). It is sensitive to outliers.
- **Median**: The middle value when data is ordered. It is "robust" because it is not heavily affected by extreme outliers.
- **Mode**: The value that appears most frequently. A dataset can be unimodal, bimodal, or multimodal.
### [[Variability]]
Measures that describe the **spread** or **dispersion** of data—how far the data points fall from the center.
- **Range**: The simplest measure; the difference between the maximum and minimum values ($Max - Min$).
- **Variance**: The average of the squared differences from the Mean. It measures the degree of "spread."
- **Standard Deviation**: The square root of the variance. It expresses the spread in the same units as the original data, making it the most commonly used metric.
- **Interquartile Range (IQR)**: The range of the middle 50% of the data ($Q_3 - Q_1$), used to identify the spread while ignoring outliers.
