# Tables

Can efficiently monitor data and implement strategies based on data insights

```python
import pandas as pd

# Sample sales data
data = {
	"Product": ["A", "B", "C", "D", "E"],
	"Sales": [15000, 22000, 18000, 24000, 20000]
}
# Creating DataFrame
df = pd.DataFrame(data)

# Generating summary statistics
summary = df["Sales"].describe()
```
    
## Outputs

```python
print("Sales Data Table:")
print(df)
```

- Displays neatly structured table displaying product names and their corresponding sales figures, making it easy to compare sales performance across different products

```python
print("Summary Statistics:")
print(summary)
```

- provides summary statistics
	- count (number of records)
	- mean (average sales)
	- standard deviation (sales variability)
	- minimum and maximum sales values
	- quartiles (percentiles that divide data into sections)

# Line Plot

```python
import matplotlib.pyplot as plt 

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [15000, 18000, 22000, 21000, 25000, 27000

plt.plot(months, sales, marker='o', linestyle='-', color='b')
plt.xlabel("Months")
plt.ylabel("Sales Revenue ($)")
plt.title("Monthly Sales Trend")
plt.show()
```
    
## Outputs

- A line chart showing monthly sales trends, allowing business managers to adjust marketing strategies for pea months

# Pie Chart

```python
labels = ["Frequent Buyers", "Occasional Buyers", "One-time Buyers"]
sizes = [45, 35, 20]
colors = ["blue", "orange", "green"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
plt.title("Customer Segmentation Based on Purchase Frequency")
plt.show()
```
## Outputs

- A pie chart displaying the percentage of customer types, helping marketing teams tailor promotions for high-value customers.

# Bar Chart

```python
employees = ["Alice", "Bob", "Charlie", "David", "Emma"]
sales = [50000, 42000, 67000, 39000, 52000]

plt.bar(employees, sales, color='teal')
plt.xlabel("Employees")
plt.ylabel("Sales ($)")
plt.title("Employee Sales Performance")
plt.show()
```
## Outputs

- A bar chart comparing employee sales performance, assisting management in recognizing high-performing employees and coaching others accordingly.

# Histogram

```python
import numpy as np 

daily_visits = np.random.normal(500, 100, 1000) # Simulating website visit data

plt.hist(daily_visits, bins=20, color="purple", edgecolor="black")
plt.xlabel("Daily Visitors")
plt.ylabel("Frequency")
plt.title("Distribution of Website Visits Over Time")
plt.show()
```
## Outputs

- A histogram depicting the distribution of website visits, helping businesses optimize ad campaigns and content strategies.

# Scatter Plot

```python
import pandas as pd

customer_ids = np.arange(1, 101)
transaction_amounts = np.random.randint(50, 5000, 100)
purchase_frequency = np.random.randint(1, 50, 100)

plt.scatter(purchase_frequency, transaction_amounts, color="red", alpha=0.5)
plt.xlabel("Purchase Frequency")
plt.ylabel("Transaction Amount ($)")
plt.title("Transaction Analysis for Fraud Detection")
plt.show()
```
## Outputs

- A scatter plot showing potential outliers, enabling fraud analysts to flag suspicious transactions for further review.

# Libraries / Tools for Visualization

## Python 

offers a variety of powerful libraries designed to create static, animated, and interactive visualizations that cater to different analytical needs. These libraries range from traditional plotting tools like Matplotlib to more sophisticated, web-based frameworks such as Plotly and Bokeh. Each library provides unique capabilities, whether for statistical analysis, real-time data exploration, or interactive dashboards. Below, we explore some of the most widely used Python visualization libraries and the types of visualizations they support.

## Matplotlib 

is one of the most widely used Python libraries for data visualization, providing a flexible and highly customizable way to generate static, animated, and interactive plots. It allows users to create a variety of visualizations, including line charts, bar charts, scatter plots, histograms, box plots, and pie charts. The library's pyplot module is particularly popular due to its similarity to MATLAB's plotting functions, making it easy to use for basic charting needs. While it offers extensive customization options, it can be complex when dealing with highly detailed visualizations. Matplotlib is often used in conjunction with other libraries like NumPy and Pandas for handling data more efficiently.

## Seaborn 

is built on top of Matplotlib and is specifically designed for statistical data visualization, making it an excellent choice for creating visually appealing and informative graphics. It provides high-level functions to create complex visualizations such as heatmaps, violin plots, swarm plots, and pair plots with minimal code. One of its key strengths is its ability to integrate well with Pandas data structures, allowing users to generate plots directly from DataFrames. Seaborn is particularly useful for visualizing relationships, distributions, and categorical data with built-in themes and color palettes that enhance readability. Its statistical functionalities, such as built-in regression plots, make it popular in data science and machine learning research.

## Plotly 

is a powerful interactive visualization library that supports a wide range of chart types, including line charts, scatter plots, bar charts, histograms, box plots, and even 3D plots. Unlike Matplotlib and Seaborn, which primarily generate static images, Plotly produces interactive graphs that allow users to zoom, pan, and hover over data points for detailed insights. It supports both offline and online plotting, making it ideal for web-based applications and dashboards. With its ability to create visually engaging charts with minimal coding, Plotly is commonly used in business intelligence, financial analysis, and scientific research. Additionally, its compatibility with Dash, a framework for building interactive web applications, extends its utility for developing dynamic data visualizations.

## Bokeh 

is a Python library designed for creating interactive and web-ready visualizations, making it an excellent choice for dashboard development and data-driven web applications. It supports various chart types, including line charts, bar charts, scatter plots, heatmaps, and network graphs, with a focus on high-performance rendering. Unlike Matplotlib, Bokeh generates visualizations using JavaScript, enabling seamless interactivity directly in web browsers. It provides multiple interfaces for different levels of complexity, from simple plots with bokeh.plotting to advanced applications using bokeh.models and bokeh.server. Its ability to handle large datasets efficiently makes it a preferred choice for real-time data visualization in analytics platforms.

## Altair

is a declarative statistical visualization library built on Vega and Vega-Lite, making it a concise and expressive tool for generating complex visualizations with minimal code. It supports a wide range of visualizations, including scatter plots, line charts, bar charts, histograms, area charts, and layered charts. One of its key strengths is its integration with Pandas, allowing users to effortlessly create visualizations from DataFrames. The declarative nature of Altair makes it easy to specify chart properties, transformations, and interactions in a structured way, improving readability and maintainability. Since it automatically handles tasks like axis scaling and legend placement, it is a great choice for users who prioritize simplicity and clarity in their visualizations.

## GGPlot

implemented in Python as _Plotnine_, is inspired by the Grammar of Graphics framework, which provides a structured way to build complex visualizations. It is particularly useful for users familiar with R’s ggplot2, offering a similar syntax for layering different graphical components such as geoms, aesthetics, and scales. Plotnine supports various chart types, including scatter plots, histograms, bar charts, line plots, and faceted plots, making it effective for exploratory data analysis. Its declarative nature enables users to create complex visualizations with concise and readable code. Although it is not as widely used as Matplotlib or Seaborn, it remains a valuable tool for users who prefer a structured approach to plotting.

## Pygal

is a lightweight Python library designed for generating interactive SVG-based visualizations with minimal code. It supports a wide range of chart types, including bar charts, line charts, pie charts, radar charts, and even more advanced plots like treemaps and maps. One of Pygal’s main advantages is its ability to generate highly customizable, scalable vector graphics (SVG), which can be embedded in web applications without losing quality. It is particularly useful for creating lightweight visualizations for web dashboards, as the generated charts are interactive and require minimal resources. Though it may not have as many advanced statistical features as Seaborn or Altair, Pygal excels in simplicity and ease of use.