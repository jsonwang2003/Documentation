# Descriptive Statistics

- ways of summarizing large sets of quantitative (numerical) information
- provides insights into
	- central tendencies
	- variability
	- patterns within datasets

- understand key characteristics of their data and identify anomalies, trends, and outliers by employing techniques such as
	- mean
	- median
	- mode
	- range
	- standard deviation
	- skewness
	- kurtosis
- serves as foundation for further data analysis
	- enable generation of actionable insights without relying on complex statistical modeling

# Statistical Measurements

## Mean
- the **arithmetic average**
- fundamental measure of central tendency
	$$
	mean = \frac{\displaystyle\sum_{i=0}^N x_i}{N}
	$$
- provides insight into the overall trend of a dataset
![Important](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAqSURBVDhPY2DAA97aWp55a2t5Bl0cGTChC5AKRg0YNYBhUBgwyAFdciMAGBcI1YEA4usAAAAASUVORK5CYII=) can be misleading when the data contains outliers or is highly skewed

```python
import numpy as np

data = [100, 200, 150, 400, 500]
mean_value = np.mean(data)
```

## Median
- the middle value in an ordered dataset
- more robust measure of central tendency when data is skewed
- less affected by extreme values
	- useful in financial and income-related analysis
```python
import numpy as np

data = [100, 200, 150, 400, 500]
median_value = np.median(data)
```

## Mode
- the value that appears most frequently in a dataset
- useful for categorical data analysis

```python
from statistics import mode

data = [42, 37, 42, 45, 42, 38, 37]
mode_value = mode(data)
```

## Range
- measure of dispersion that represents the difference between the maximum and minimum values in a dataset
- provides simple measure of variability
- does not account for how data is distributed between these extremes
```python
data = [42, 37, 42, 45, 42, 38, 37]
range_value = max(data) - min(data)
```

## Standard Deviation
- quantifies the amount of variation or dispersion in a dataset
	- Lower standard deviation: close to the mean
	- Higher standard deviation: greater variability
```python
import numpy as python

std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
```

## Skewness
- measures the asymmetry of a distribution relative to its mean
	- positive skew: long right tail
	- negative skew: long left tail
```python
from scipy.stats import skew

skewness_value = skew(data)
```

## Kurtosis
- measures the "tailedness" of a distribution, indicating whether data points are concentrated around the mean or dispersed across the tails

```python
from scipy.stats import kurtosis

kurtosis_value = kurtosis(data)
```