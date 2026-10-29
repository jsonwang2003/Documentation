## 1. 
For which of the following random variables do you think the mean might be significantly different from the median? Give a brief justification in each case; there is subjectivity here, so what matters is your reasoning.

### (a) 
Pick a person at random from a big city (e.g., New York), and let **H** denote their height. 

**Answer**
The mean will not be significantly different from the median since the distribution of people's height should be even with low probability to find someone exceedingly tall or short.
### (b) 
Let **C** denote the cost of their house. 

**Answer**
The cost of a person's house depends on multiple factors, which the distribution of the price range skews right with all the luxurious housing, making the median likely to be lower than the mean of the overall housing market.
### (c) 
Let **G** denote their high school GPA. 

**Answer**
The GPA is bounded within the range $[0, 5]$, the mean should be tending towards the middle, making the gap between the mean and the median not significantly different.
### (d) 
Let **S** denote their salary.

**Answer**
Since salary of an individual heavily depends on one's job position with significant growth as one's position goes up, the graph of the distribution will be right-skewed, making the gap between the mean and the median be significantly different.

---
## 2. 
A random variable **X** has the following distribution:

- $Pr(X=1)=\frac{1}{4}$
- $Pr(X=2) = \frac{1}{8}$
- $Pr(X=3)=\frac{1}{2}$
- $Pr(X=4)=\frac{1}{8}$

### (a) 
What is the **mean** of X? 

**Answer**
$$
\begin{align*}
\mathbb{E}(X) &= \Sigma_{i=1}^4 (x \times Pr(x))\\
&= 1\left( \frac{1}{4} \right) + 2\left( \frac{1}{8} \right) + 3\left( \frac{1}{2} \right) + 4\left( \frac{1}{8} \right)\\
&= \frac{2}{8} + \frac{2}{8}+ \frac{12}{8} + \frac{4}{8}\\
&= \frac{20}{8}\\
&= \boxed{2.5}
\end{align*}
$$
### (b) 
What is the **median** of X?

**Answer**
The median of $X$ = m such that $Pr(X\geq m) \geq \frac{1}{2}$ and $Pr(X\leq m) \geq \frac{1}{2}$
1. $Pr(X  \leq 1) = \frac{1}{4}$
2. $Pr(X\leq2) = \frac{1}{4} + \frac{1}{8} = \frac{3}{8}$
3. $Pr(X \leq 3) = \frac{3}{8} + \frac{1}{2} = \frac{7}{8}$

Since $X=2$ does not satisfy the condition, the median of $X$ is $3$

---
## 3. 
A coin is tossed 50 times; of these, 30 are heads (1’s). What is the empirical distribution $\hat{P}$?

**Answer**
- $\hat{P}(\text{head}) = \frac{30}{50} = \frac{3}{5}$
- $\hat{P}(\text{tail}) = \frac{20}{50} = \frac{2}{5}$

---
## 4. 
Two random variables $X,Y \in \{ 1,2,3 \}$ have joint distribution:

|         | **Y=1** | **Y=2** | **Y=3** |
| ------- | ------- | ------- | ------- |
| **X=1** | 0.1     | 0.2     | 0.05    |
| **X=2** | 0.1     | 0.1     | 0.1     |
| **X=3** | 0.1     | 0.2     | 0.05    |

### (a) 
Are X and Y independent? 

**Answer**
Two random variables are independent if $Pr(X=x, Y=y) = Pr(X=x) \times Pr(Y=y)$

Probabilities of $X$ and $Y$

| X   | Pr(X) | Y   | Pr(Y) |
| --- | ----- | --- | ----- |
| 1   | 0.35  | 1   | 0.3   |
| 2   | 0.3   | 2   | 0.5   |
| 3   | 0.35  | 3   | 0.2   |
at $Pr(X=1, Y=2)$, the joint probability is $0.2$, but the product of the individual probabilities $Pr(X=1)Pr(Y=2) = 0.175$. Therefore, $X$ and $Y$ are not independent
### (b) 
What is the correlation between X and Y?

**Answer**
- $\mathbb{E}(X) = 1(.35) + 2(.3) + 3(.35) = 0.35 + 0.6 + 1.05 = \boxed{2}$
- $\mathbb{E}(Y) = 1(.3) + 2(.5) + 3(.2) = 0.3 + 1 + 0.6 = \boxed{1.9}$
$$
\begin{align*}
\mathbb{E}(XY) &= \sum_{x=1}^3\sum_{y=1}^{3} xy \cdot Pr(X=x, Y=y) \\
&= (1)(1)(0.1) + (1)(2)(0.2) + (1)(3)(0.05)\\
&+(2)(1)(0.1) + (2)(2)(0.1) + (2)(3)(0.1)\\
&+ (3)(1)(0.1) + (3)(2)(0.2) + (3)(3)(0.05)\\
&= 0.65 + 1.2 + 1.95 = \boxed{3.8}
\end{align*}
$$

Since $\mathbb{E}(XY) = 3.8 = \mathbb{E}(X)\mathbb{E}(Y)$, the random variables $X$ and $Y$ are uncorrelated

---
## 5. 
Let $X$ be any real-valued random variable that takes in $\mathbb{R}$, and let $Y=aX+b$ for some constants $a,b$.

### (a) 
Give a formula for the covariance between $X$ and $Y$ in terms of the variance of $X$.

**Answer**
$$
	\begin{align*}
	cov(X,Y) &= cov(X,aX+b)\\
	&= a \cdot cov(X, X+b)\\
	&= a \cdot cov(X,X)\\
	&= \boxed{a \cdot var(X)}
	\end{align*}
$$
### (b) 
What is the correlation between $X$ and $Y$?

**Answer**
- $\sigma_{Y} = \sqrt{ Var(Y) } = \sqrt{ Var(aX+b) } = \sqrt{ a^{2}Var(X) } = |a|\sigma_{X}$
$$
	\begin{align*}
	\text{corr}(X,Y) &= \frac{cov(X,Y)}{std(X)std(Y)}\\
	&= \frac{a \cdot Var(X)}{\sigma_{X}(|a|\sigma_{X})}\\
	&= \frac{a\cdot Var(X)}{|a|Var(X)}\\
	&= \boxed{\frac{a}{|a|}}
	\end{align*}
$$

---
## 6. 
Consider the dataset of five points in R1:

$$
	-10, -8, 0, 8, 10
$$

Cluster into $k=3$ groups.

### (a) 
What is the **optimal k-means solution**? Give the center locations and the k-means cost. 

**Answer**
- $\mu_{1} = \frac{-10+ -8}{2} = -9$
- $\mu_{2}=\frac{0}{1}=0$
- $\mu_{3}=\frac{8+10}{2}=9$

$$
	\begin{align*}
	\text{cost} &= \sum_{i=1}^n \min_{j} |x_{i}-\mu_{j}|^{2}\\
	&= (-10 - (-9))^{2} + (-8 - (-9))^{2} + (0-0)^{2} + (8-9)^{2} + (10-9)^{2}\\
	&= 1 + 1 + 0 + 1 + 1\\
	&= \boxed{4}
	\end{align*}
$$
### (b) 
Suppose we call Lloyd’s k-means algorithm on this data, with $k = 3$ and with initialization $\mu_{1} = −10,\mu_{2} = −8,\mu_{3} = 0$. What is the final set of cluster centers obtained by the algorithm? What is the k-means cost of this set of centers?

**Answer**
- $-10$, nearest center is $-10$, so assign to $-10$
- $-8$, nearest center is $-8$, so assign to $-8$
- $0$, nearest center is $0$, so assign to $0$
- $8$, nearest center is $0$, so assign to $0$, center $0$ averaged towards $\frac{0 + 8}{2} = 4$
- $10$, nearest center is $4$, so assign to $4$, center $4$ averaged towards $\frac{0 + 8 + 10}{3} = 6$

Final Set of Cluster Centers: $\{ -10, -8, 6 \}$

$$
	\begin{align*}
	Cost &= \sum_{i=1}^{n}\min_{j}|x_{i} - \mu_{j}|^{2}\\
	&= (-10 - (-10))^{2} + (-8 - (-8))^{2} + (0 - 6)^{2} + (8 - 6)^{2} + (10-6)^{2}\\
	&= 0 + 0 + 36 + 4 + 16\\
	&= \boxed{56}
	\end{align*}
$$

---
## 7. 
The k-means algorithm relies on the following basic result: if we have a cluster consisting of points $x_{1},...,x_{n}\in\mathbb{R}^{d}$ and we want to choose a "center" $v\in\mathbb{R}^{d}$ so as to minimize the total squared Euclidean distortion

$$ \sum_{i=1}^{n}||x_{i}-v||_{2}^{2} $$

then the optimal choice is to set $v$ to the mean of the points, that is, $(1/n)\sum_{i}x_{i}$. In this problem, we will see why this is true.

### (a) 
First let's do this in the one-dimensional case (that is, $d=1$). Show that the value $v\in\mathbb{R}$ that minimizes

$$ L(v)=\sum_{i=1}^{n}(x_{i}-v)^{2} $$

is the mean of the $x_{i}$'s. One way to do this is by calculus (take the derivative $L^{\prime}(v)$ and set it to zero). Instead, use a different argument: show by algebraic manipulation that if $\mu = mean(x_{1},...,x_{n})$, then for any u,

$$ \sum_{i}(x_{i}-v)^{2}=\sum_{i}(x_{i}-\mu)^{2}+n(\mu-v)^{2}. $$

The nice thing about this expression is that it not only shows that the optimal choice is $v=\mu$, but it also quantifies the additional cost of choosing a different v.

**Answer**

$$
	\begin{align*}
	L(v) &= \sum_{i=1}^{n}(x_{i}-v)^{2}\\
	&= \sum_{i=1}^{n}(x_{i} - \mu + \mu - v)^{2}\\
	&= \sum_{i=1}^{n}[(x_{i} - \mu)^{2} + 2(x_{i}-\mu)(\mu-v) + (\mu-v)^{2}]\\
	&= \sum_{i=1}^{n}(x_{i}-\mu)^{2} + \sum_{i=1}^{n} 2(x_{i}-\mu)(\mu-v) + \sum_{i=1}^{n}(\mu-v)^{2}\\
	&= \sum_{i=1}^{n}(x_{i}-\mu)^{2} + 2(\mu-v)\sum_{i=1}^{n} (x_{i}-\mu) + \sum_{i=1}^{n}(\mu-v)^{2}\\
	&= \sum_{i=1}^{n}(x_{i} - \mu)^{2} + \sum_{i=1}^{n}x_{i} - \sum_{i=1}^{n}\mu + n(\mu-v)^{2} \\ \mu = &\text{mean of the points} = \frac{1}{n}\sum_{i=1}^{n}x_{i}\\
	&= \sum_{i=1}^{n}(x_{i} - \mu)^{2} + n\mu - n\mu + n(\mu-v)^{2}\\
	&= \sum_{i=1}^{n}(x_{i} - \mu)^{2} + 0 + n(\mu-v)^{2}\\
	\sum_{i=1}^{n}(x_{i} - v)^{2} &= \sum_{i=1}^{n} (x_{i} - \mu)^{2} + n(\mu-v)^{2} 
	\end{align*}
$$

### (b) 
Can you generalize your argument in (a) to higher-dimensional points, $r_{1},...,x_{n}\in\mathbb{R}^{d}$? This is an interesting relation: it exactly captures the squared distortion induced by using a location parameter other than the mean.

**Answer**
Let $x_{1}, x_{2}, \dots, x_{n}, \mu, v \in  \mathbb{R}^{d}$, where the mean vector is $\mu = \frac{1}{n}\sum_{i=1}^{n}x_{i}$

$$
	\begin{align*}
	\sum_{i=1}^{n}||x_{i} - v||^{2}_{2} &= \sum_{i=1}^{n}||(x_{i}-\mu) + (\mu - v)||^{2}_{2}\\
	&= \sum_{i=1}^{n}[||x_{i} - \mu||^{2}_{2} + 2(x_{i} - \mu)^{T}(\mu-v) + ||\mu - v||^{2}_{2}]\\
	&= \sum_{i=1}^{n}||x_{i} - \mu||^{2}_{2} + 2\left( \sum_{i=1}^{n}(x_{i} - \mu) \right)^{T}(\mu-v) + \sum_{i=1}^{n}||\mu-v||^{2}_{2}\\ \\
	\sum_{i=1}^{n}(x_{i}-\mu) &= \sum_{i=1}^{n}x_{i} - n\mu = n\mu - n\mu = \vec{0}\\ \\
	&= \sum_{i=1}^{n}||x_{i} - \mu||^{2}_{2} + 2\left( \vec{0} \right)^{T}(\mu-v) + \sum_{i=1}^{n}||\mu-v||^{2}_{2}\\
	&= \sum_{i=1}^{n}||x_{i} - \mu||^{2}_{2} + \sum_{i=1}^{n}||\mu-v||^{2}_{2}\\
	\sum_{i=1}^{n}||x_{i} - v||^{2}_{2} &= \sum_{i=1}^{n}||x_{i} - \mu||^{2}_{2} + n||\mu-v||^{2}_{2}
	\end{align*}
$$

---
## 8. 
Give an example of three points $x_{1},x_{2},x_{3} \in \mathbb{R}$ such that the value $v \in \mathbb{R}$ that minimizes:

$$
	L(v) = \sum_{x}|x_{i} - v|
$$

is **not** the mean of the points.

_(Hint: The minimizing value is any median.)_

**Answer**
An example is the following 1-dimensional points $x_{1}=0, x_{2}=1$ and $x_{3}=5$

- $\text{mean: }\mu = \frac{0+1+5}{3} = \frac{6}{3} = 2$
- $\text{median: } m=1$

$$
\begin{align*}
L(\mu) = L(2) &= |0-2| + |1-2| + |5-2| \\
&= 2 + 1 + 3 = \boxed{6}
\end{align*}
$$

$$
	\begin{align*}
	L(m) = L(1) &= |0-1|+|1-1|+|5-1|\\
	&= 1 + 0 + 4 = \boxed{5}
	\end{align*}
$$

Since $L(1) = 5 < 6 = L(2)$, the value $v$ that minimizes the $\mathcal{l}_{1}$ distance is the median $m=1$, not the mean $\mu = 2$

---
## 9. 

The file `data2020.xls` contains 181 countries with:
- GDP per capita (2020, USD)
- Infant mortality per 1000 (2020)
### (a) 
Load the data and print histograms of both attributes. 

**Answer**
```python
df = pd.read_excel("Data_file_data2020.xls")
pd.set_option('display.max_rows', None)
countries = df.iloc[:, 0].to_list()
X = df.iloc[:, 1:].to_numpy()
```
### (b) 
Show a 2D plot of GDP vs. infant mortality.

**Answer**
```python
plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], alpha = 0.7)
plt.title('GDP per capita vs. infant Mortality')
plt.xlabel('GDP per capita (US Dollars)')
plt.ylabel('Infant mortality per 1000')
plt.show()
```

![[Figure_1.png]]
### (c) 
Normalize each feature to mean 0 and variance 1, then run k-means with k=5. Show the color-coded plot. 

**Answer**
```python
scalar = StandardScaler()
X_normalized = scalar.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_normalized)

plt.figure(figsize=(10,6))
scatter = plt.scatter(X_normalized[:, 0], X_normalized[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering (k=5) on Normalized Data')
plt.xlabel('Normalized GDP per capita')
plt.ylabel('Normalized Infant mortality')
plt.colorbar(scatter, label="Cluster ID")
plt.show()
```

![[Figure_2.png]]
### (d) 
List the countries in each cluster. 
Cluster 0: Afghanistan, Chad, Equatorial Guinea, Guinea, Lesotho, Liberia, Mali, Niger, Nigeria, Pakistan, Sierra Leone, Somalia, Fed. Rep.

**Answer**
```python
cluster_dict = {i:[] for i in range(5)}
for country, cluster_id in zip(countries, clusters):
	cluster_dict[cluster_id].append(country)

for cluster_id in range(5):
    print(cluster_id, ", ".join(cluster_dict[cluster_id]))
```

Cluster 1: Australia, Austria, Belgium, Canada, Denmark, Finland, France, Germany, Iceland, Ireland, Israel, Luxembourg, Netherlands, New Zealand, North America, Norway, OECD members, Qatar, Singapore, Sweden, Switzerland, Turks and Caicos Islands, United Arab Emirates, United Kingdom, United States

Cluster 2: Antigua and Barbuda, Argentina, Bahamas, The, Bahrain, Barbados, Belarus, Brunei Darussalam, Bulgaria, Chile, China, Costa Rica, Croatia, Cyprus, Czechia, Estonia, Euro area, Greece, Hungary, Italy, Japan, Kazakhstan, Korea, Rep., Kuwait, Latvia, Lithuania, Malaysia, Maldives, Malta, North Macedonia, Oman, Panama, Poland, Portugal, Romania, Russian Federation, Saudi Arabia, Seychelles, Slovak Republic, Slovenia, Spain, St. Kitts and Nevis, Turkiye, Uruguay

Cluster 3: Albania, Algeria, Armenia, Azerbaijan, Bangladesh, Belize, Bhutan, Bolivia, Bosnia and Herzegovina, Brazil, Cabo Verde, Cambodia, Colombia, Cuba, Ecuador, Egypt, Arab Rep., El Salvador, Fiji, Georgia, Grenada, Guatemala, Guyana, Honduras, India, Indonesia, Iran, Islamic Rep., Iraq, Jamaica, Jordan, Kyrgyz Republic, Lebanon, Libya, Marshall Islands, Mauritius, Mexico, Micronesia, Fed. Sts., Moldova, Mongolia, Morocco, Nepal, Nicaragua, Palau, Paraguay, Peru, Philippines, Samoa, Sao Tome and Principe, Solomon Islands, Sri Lanka, St. Lucia, St. Vincent and the Grenadines, Suriname, Tajikistan, Thailand, Tonga, Tunisia, Ukraine, Uzbekistan, Vanuatu, Venezuela, RB, Viet Nam

Cluster 4: Angola, Benin, Botswana, Burkina Faso, Burundi, Cameroon, Comoros, Congo, Dem. Rep., Congo, Rep., Cote d'Ivoire, Djibouti, Dominica, Dominican Republic, Eswatini, Ethiopia, Gabon, Gambia, The, Ghana, Guinea-Bissau, Haiti, Kenya, Kiribati, Lao PDR, Madagascar, Malawi, Mauritania, Mozambique, Myanmar, Namibia, Papua New Guinea, Rwanda, Senegal, Sudan, Tanzania, Timor-Leste, Togo, Turkmenistan, Uganda, Zambia, Zimbabwe
### (e) 
What do you think was the point of normalizing the data in (c)?

**Answer**
I think the purpose behind normalizing the data in (c) was due to the disproportional difference between GDP and Infant mortality. The GDP feature ranges in thousands of dollars whereas infant mortality is measured in a much smaller scale. By normalizing the two features, the Euclidean distance which k-means uses, will not be skewed and both GDP and infant mortality statistic will contribute to the analysis equally