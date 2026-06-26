## 1. 
When the line current is held at 10 amps for a speed of 1,500 rpm, the true mean stray-load loss for a certain type of induction motor is (watts). Assume that stray-load loss is normally distributed with $\sigma=3.0$

### (a) (10 points) 
Compute an 82% confidence interval for when $n=100$ and $\overline{X}=58.3.$ Interpret the confidence interval in the context of the problem.

**Solution:**

1. Parameter of Interest: $\theta = \mu$
2. Sample Statistic: $\hat{\theta} = \overline{X} = 58.3$
3. Sampling Distribution: $\frac{\theta - \hat{\theta}}{SE(\hat{\theta})} \sim N(\mu, \frac{\sigma^2}{n})$
4. Standard Error: $SE(\hat{\theta}) = \frac{\sigma}{\sqrt{n}} = \frac{3}{\sqrt{100}} = \frac{3}{10} = 0.3$
5. $0.82 = 1- \alpha, \alpha = 0.18$, $\frac{\alpha}{2} \text{quantile} = q_{\alpha/2} = Z_{\alpha / 2} = 1.34$

![[Pasted image 20260307140921.png]]

6. Confidence Interval: 
$$
\begin{align*}
CI(\alpha, \theta) &= \big[ \hat{\theta} - q_{\alpha/2}\cdot SE(\hat{\theta}), \hat{\theta} + q_{\alpha/2}\cdot SE(\hat{\theta})\big]\\
&= \big[ (58.3) - (1.34)(0.3), (58.3) + (1.34)(0.3)\big]\\
&= \boxed{[57.898, 58.702]}
\end{align*}
$$

Interpretation: 
We are $82\%$ confident that the true mean stray-load loss for this type of induction motor is between $57.898$ and $58.702$ watts.
### (b) (10 points) 
How large must $n$ be if the width of the 99% confidence interval for is required to be 1.0?

**Solution:**

let $z_{\alpha/2}\cdot SE(\hat{\theta}) = 0.5$

$$
\begin{align*}
\alpha &= 1-0.99 = 0.01\\
z_{\alpha/2} &= 2.58\\
\end{align*}
$$
![[Pasted image 20260307141858.png|571]]

$$
\begin{align*}
0.5 &= z_{\alpha/2}\cdot SE(\hat{\theta})\\
&= (2.58)(\frac{\sigma}{\sqrt{n}})\\
0.1938 &= \frac{3}{n}\\
\frac{\sqrt{n}}{3} &= 5.16\\
\sqrt{n} &= 15.48\\
n &= 239.6 \approx \boxed{240}
\end{align*}
$$

---
## 2. (20 points) 
A journal article reports that a sample of size $n=5$ was used as a basis for calculating a 95% confidence interval for the true mean natural frequency (Hz) of delaminated beams of a certain type. The resulting confidence interval was $[229.764, 233.504]$. You decide that a confidence level of 99% is more appropriate than the 95% level used. What are the limits of the 99% interval? Assume that the data collected follows a Normal distribution.

**Solution:**

First find all information required that can be extracted by the given Confidence interval, sample size, and $\alpha$

$\alpha' = 1-0.95 = 0.05$
$q_{\alpha'/2} = z_{\alpha'/2} = 1.96$

![[Pasted image 20260307142224.png]]

$\hat{\theta} = \frac{l_{\alpha'} + u_{\alpha'}}{2} = \frac{229.764 + 233.504}{2} = \frac{463.268}{2} = 231.634$



$$
\begin{align*}
ME(\alpha', \theta) = q_{\alpha/2} \cdot SE(\hat{\theta}) &= u_{\alpha'} - \hat{\theta}\\
1.96 \cdot SE(\hat{\theta}) &= 233.504 - 231.634\\
1.96 \cdot SE(\hat{\theta}) &= 1.87\\
SE(\hat{\theta}) &= 0.9541
\end{align*}
$$

Next, compute the Confidence Interval if wanting $99\%$ confidence:

$q_{\alpha/2} = z_{\alpha/2} = 2.58$

![[Pasted image 20260307144125.png]]

$$
\begin{align*}
\alpha = 1-0.99 &= 0.01\\
CI(\alpha, \theta) &= \big[\hat{\theta} - q_{\alpha/2} \cdot SE(\hat{\theta}), \hat{\theta} + q_{\alpha/2}\cdot SE(\hat{\theta}) \big]\\
&= \big[(231.634) - (2.58)(0.9541), (231.634) + (2.58)(0.9541)\big]\\
CI(\alpha, \theta) &= [229.172, 234.096]
\end{align*}
$$

---

## 3. (20 points) 
A sample of 66 adults were put on a low-carbohydrate diet for a year. The average weight loss was 11 lbs and the standard deviation was 19 lbs. Compute an upper 99% confidence interval for the true average weight loss. What does the interval say about our confidence that the mean weight loss is positive?

**Solution:**

- $n = 66$
- $\mu = 11$
- $\sigma = 19$
- $\alpha = 0.01$
- $q_\alpha = z_\alpha = 2.33$
![[Pasted image 20260307144730.png]]

$$
\begin{align*}
CI_{upper}(\alpha, \theta) &= [l_\alpha, \infty)\\\\
l_\alpha &= \hat{\theta} - q_\alpha \cdot SE(\hat{\theta})\\
&= 11 - 2.33 \cdot \frac{19}{\sqrt{66}}\\
&= 11 - 5.45\\
l_\alpha &= 5.55\\\\
CI_{upper}(\alpha, \theta) &= [5.55, \infty)
\end{align*}
$$
Since the entire lower confidence interval $[5.555, \infty)$ is above zero, we are 99% confident that the mean weight loss is positive.

---
## 4. 
A survey was conducted on $n=200$ people at UCSD where people were asked whether or not they use TikTok. The researchers found the (two-sided) 93% confidence interval for $p$ to be $CI(0.07,p)=[0.61,0.67]$. Using this information:

### (a) (10 points) 
Compute an upper 93% confidence interval for the population proportion p.

**Solution:**

What we know from the given information:
- $n = 200$
- $\alpha = 0.07$
- $q_{\alpha/2} = 1.81$

![[Pasted image 20260307150407.png]]

- $\hat{\theta} = \frac{0.61+0.67}{2} = 0.64$
$$
\begin{align*}
q_{\alpha/2} \cdot SE(\hat{\theta}) &= \hat{\theta} - l_{\alpha/2}\\
1.81 \cdot SE(\hat{\theta}) &= 0.64-0.61\\
SE(\hat{\theta}) &= \frac{0.03}{1.81}\\
SE(\hat{\theta}) &= 0.0166
\end{align*}
$$

Use the given information, we can find:
- $q_\alpha = 1.48$

![[Pasted image 20260307150437.png]]

Therefore the upper confidence interval can be found as the following
$$
\begin{align*}
CI_{upper}(\alpha, \theta) &= \big[\hat{\theta} - q_{\alpha}\cdot SE(\hat{\theta}), \ \infty\big)\\
&= [0.64 - 1.48 \cdot 0.0166,\ \infty)\\
&= [0.615, \ \infty)
\end{align*}
$$

Since we are finding the proportion, the ultimate bound will be:

$$
\boxed{CI_{upper}(\alpha, \theta) = [0.615, 1]}
$$

### (b) (10 points) 
Compute a lower 93% confidence interval for the population proportion p.

**Solution:**

Using the same given information we found from part a:

$$
\begin{align*}
CI_{lower}(\alpha, \theta) &= \big( -\infty, \ \hat{\theta} + q_{\alpha} \cdot SE(\hat{\theta}) \big]\\
&= (-\infty, \ 0.64 + 1.48 \cdot 0.0166]\\
&= (-\infty, \ 0.665)
\end{align*}
$$

Since we are finding the proportion, the ultimate bound will be:

$$
\boxed{CI_{lower}(\alpha, \theta) = [0, 0.665]}
$$
### (c) (5 points) 
Suppose the researchers wanted to make the case that: "with 93% confidence at least $x$ fraction of the UCSD student population uses Tik Tok" where $x$ is some number between 0 and 1. Which of the three types of confidence intervals (upper, lower, or two-sided) is appropriate?

**Solution:**

The upper confidence interval is more appropriate

### (d) (5 points) 
Provide an interpretation for the confidence interval you chose in part (c). Why?

**Solution:**

This interval provides a minimum value (lower bound) that we are 93% confident the true population proportion $p$ exceeds. Since the upper confidence interval encapsulates the range from lower bound to $1$

---

## 5. (12 points) 
Let $X_{1},X_{2},...,X_{n} \overset{iid}{\sim} N(\mu_{X},\sigma_{x}^{2})$ and let $Y_{1},Y_{2},...,Y_{m} \overset{iid}{\sim} N(\mu_{Y},\sigma_{Y}^{2}).$ For the following pairs of assertions: 1. Indicate whether they constitute a valid hypothesis test, and 2. Why they do (or don't) constitute a valid hypothesis test

**Solution:**
* $H_{0}:\mu_{Y}=100$ VS. $H_{a}:\overline{Y} \ne 100$
	* This does not constitute a valid hypothesis test
	* The alternate hypothesis is based on sample mean instead of true mean
* $H_{0}:\mu_{X}=100$ VS. $H_{a}:\mu_{Y}<100$
	* This does not constitute a valid hypothesis test
	* The null hypothesis is not disjoint from the alternative hypothesis
* $H_{0}:\mu_{X}=100$ vs. $H_{a}:\mu_{X}>100$
	* This does constitute a valid hypothesis test
	* Both hypotheses refer to the same population parameter; The null hypothesis is disjoint from the alternative hypothesis
* $H_{0}:max(X_{1},X_{2},...,X_{n})=100$ vs. $H_{a}:max(X_{1},X_{2},...,X_{n})<100$
	* This does not constitute a valid hypothesis test
	* $max(X_1, X_2, ..., X_n)$ is a sample statistic, which is a random variable
* $H_{0}:p \ne 0.25$ vs. $H_{a}:p=0.25$
	* This does not constitute a valid hypothesis test
	* The null hypothesis set is bigger than the alternative hypothesis set
* $H_{0}:\mu_{X}-\mu_{Y}=25$ vs. $H_{a}:\mu_{X}-\mu_{Y}>100$
	* This does not constitute a valid hypothesis test
	* There is a gap that the alternative hypothesis fail to capture $(25, 100]$


---
## 6. 
For each of the following scenarios: write down your assumptions about the distribution of the data, then write down the null hypotheses $H_{0}$ and the alternate hypotheses $H_{a}$ which enable testing the main question of interest, e.g., $X_{1},X_{2},...,X_{n} \overset{iid}{\sim} D(\theta)$ and $H_{0}:\theta=\theta_{0}$ vs. $H_{a}:\theta \ne \theta_{0}$.

### (a) (5 points) 
A school counselor believes that less than 60% of students participate in extracurricular activities. To test this hypothesis, she conducts a survey of a random sample of 200 students in the school, asking whether they participate in any extracurricular activities.

**Solution:**

Each student either participates in extracurricular activities or not, therefore the distribution of the data is:

$$
X_1, X_2, ..., X_n \overset{iid}{\sim} Ber(p)
$$
- $H_0: p = 0.6$
- $H_a: p < 0.6$

### (b) (5 points) 
A researcher wants to know if the proportion of students who own a smartphone is different between high school and middle school students. He collects data by randomly sampling 150 high school students and 150 middle school students and asks them whether they own a smartphone.

**Solution:**

Each student either owns a smartphone or not, with 2 groups:
$$
\begin{align*}
X_1, X_2, ..., X_{150} &\overset{iid}{\sim} Ber(p_x)\\
Y_1, Y_2, ..., Y_{150} &\overset{iid}{\sim} Ber(p_y)
\end{align*}
$$

With $X$ representing the response of $150$ high school students, and $Y$ representing the response of $150$ middle school students

- $H_0: p_x - p_y = 0$
- $H_a: p_x - p_y \ne 0$ 
### (c) (5 points) 
A comprehensive national survey found that people read, on average, 5 books every 3 months. A skeptical teacher thinks that this is an over-estimate when it comes to the college student subpopulation. So, she collects data by asking every student in her class to report the number of books they have read during the fall quarter.

**Solution:**

Since the statistic wants to find the mean of the books read in a fixed time interval, we use Poisson Distribution for the class survey:

$$
X_1, X_2, ..., X_n \overset{iid}{\sim} Poi(\lambda)
$$
- $H_0: \lambda = 5$
- $H_a: \lambda < 5$
### (d) (5 points) 
A nutritionist believes that the average daily calorie intake of teenagers is more than 500 calories higher than that of children. To test this hypothesis, she conducts a study where she asks a random sample of 100 teenagers and 120 children to record their daily calorie intake for a week.

**Solution:**

Since the statistic is finding the average in two groups to test the difference, we use 2 groupings, both with Normal Distributions:

$$
\begin{align*}
X_1, X_2, ..., X_{100} &\overset{iid}{\sim} N(\mu_x, \sigma^2_x)\\
Y_1, Y_2, ..., Y_{120} &\overset{iid}{\sim} N(\mu_y, \sigma^2_y)
\end{align*}
$$
Where $X$ represents the calorie intake of a teenager and $Y$ represents the calorie intake of a child

- $H_0 : \mu_x - \mu_y = 500$
- $H_a: \mu_x - \mu_y > 500$

---
## 7. 
For the hypothesis testing of the population mean, suppose the test statistic $\hat{T}$ has a standard normal $N(0,1)$ distribution when $H_{0}$ is true. Calculate the Type-I error probability $\alpha$ for each of the following situations:

### (a) (5 points) 
$H_{a}:\mu>\mu_{0}$, and the rejection region is $R(\alpha,T)=(1.88,\infty)$

**Solution:**

$$
\mathbb{P}(Z > 1.88) = \boxed{0.0301}
$$

![[Pasted image 20260307161941.png]]

### (b) (5 points)
$H_{a}:\mu<\mu_{0}$, and the rejection region is $R(\alpha,T)=(-\infty,-2.75]$

**Solution:**

$$
\begin{align*}
\mathbb{P}(Z < -2.75) &= \boxed{0.003}
\end{align*}
$$

![[Pasted image 20260307162004.png]]
### (c) (5 points)
$H_{a}:\mu \ne \mu_{0}$ and the rejection region $R(\alpha,T)=(-\infty,-2.88] \cup (2.88,\infty)$

**Solution:**

$$
\begin{align*}
\mathbb{P}(Z < -2.88) + \mathbb{P}(Z > 2.88) &= 1 - \mathbb{P}(-2.88 \le Z \le 2.88)\\ &= 1 - 0.996\\
&= 0.004
\end{align*}
$$

![[Pasted image 20260307162531.png]]

---
## 8. 
The melting point of each of $n=16$ samples $X_{1},X_{2},...,X_{n}$ of a brand of hydrogenated vegetable oil was determined, resulting in $\overline{X}=94.32.$ Assume that the distribution of $X_{1},X_{2},...,X_{n}$ is $N(\mu,1.20^{2})$ normal with known $\sigma=1.20$.

### (a) (5 points) 
For the hypotheses $H_{0}:\mu=95$ vs. $H_{a}:\mu \ne 95$ calculate the rejection region $R(\alpha,\theta)$ at level $\alpha=0.01$.

**Solution:**

$$
R(\alpha, \theta) = (-\infty, \ \theta_0 - q_{\alpha/2}\cdot SE]\ \cup [\theta_0 + q_{\alpha/2} \cdot SE,\ \infty)
$$
- $\theta_0 = \mu_0 = 95$
- $q_{\alpha/2} = 2.58$
- $SE = \frac{\sigma}{\sqrt{n}} = \frac{1.2}{\sqrt{16}} = 0.3$

$$
\begin{align*}
R(\alpha, \theta) &= (-\infty, \ \theta_0 - q_{\alpha/2}\cdot SE]\ \cup [\theta_0 + q_{\alpha/2} \cdot SE,\ \infty)\\
&= (-\infty, \ 95 - 2.58\cdot 0.3]\ \cup [95 + 2.58 \cdot 0.3,\ \infty)\\
&= (-\infty, \ 94.226]\ \cup [95.774,\ \infty)
\end{align*}
$$
### (b) (5 points) 
For the same test with $\alpha=0.01$ test, what is the probability of a Type-II error when the true $\mu=94$ under $H_{a}$?

**Solution:**
- $\mu_a = \theta_a = 94$

$$
\begin{align*}
P(\text{Type-II}) &= P(\text{fail to reject }H_0 | H_a \text{ true})\\
&= P(\theta_0 - q_{\alpha/2}\cdot SE < \hat{\theta} < \theta_0 + q_{\alpha/2} \cdot SE)\\
&= P(\frac{\theta_0 - \theta_a}{SE} - q_{\alpha/2} < \frac{\hat{\theta} - \theta_a}{SE} < \frac{\theta_0 - \theta_a}{SE} + q_{\alpha/2})\\
&= P(\frac{95-94}{0.3} - 2.58 < Z < \frac{95-94}{0.3} + 2.58)\\
&= P(0.75 < Z < 5.91)\\
&= P(Z < 5.91) - P(Z < 0.75)\\
&= 1 - 0.7734\\
P(\text{Type-II})&= \boxed{0.2266}
\end{align*}
$$

### (c) (5 points)
Compute the p-value of the two-tailed hypothesis test, and conclude whether this p-value agrees with your conclusion from part (a).

**Solution:**
$$
\begin{align*}
\text{p-value} &= 2 \times P(x > |\hat{T}|)\\
&= 2 \times P(x > |\frac{\overline{X} - \mu_0}{SE}|)\\
&= 2 \times P(x > |\frac{94.32 - 95}{0.3}|)\\
&= 2 \times P(x > |-2.267|)\\
&= 2 \times P(x > 2.267)\\
&= 2 \times 0.0117\\
&= \boxed{0.0234}
\end{align*}
$$

From part (a), the mean is outside of the rejection region, therefore part (a) will conclude it failed to reject null hypothesis.

From part (c), the $\text{p-value} = 0.0234 > 0.01 = \alpha$, therefore part (c) will also conclude it failed to reject null hypothesis as p-value is the smallest rate $\alpha$ can be to reject $H_0$

In conclusion, this p-value agrees with the conclusion from part a

---
## 9. 
In a study to assess cardiovascular health, researchers measured heart rate recovery after moderate exercise. For $n=10$ athletes and $m=11$ non-athletes, the summary statistics for the average recovery rate (measured as decrease in beats/minute in a five minute window) is given in Table 1.

| Sample | Number of samples | Sample Mean ($\overline{X}$) | Sample Variance ($\hat{\sigma}^{2}$) |
| :--- | :--- | :--- | :--- |
| Athletes | $n=10$ | $\overline{X}=0.64$ | $\hat{\sigma_{X}}^{2}=0.2$ |
| Non-athletes | $m=11$ | $\overline{Y}=2.05$ | $\hat{\sigma_{Y}}^{2}=0.4$ |

*Table 1: Cardiovascular data summary*

### (a) (3 points) 
Consider testing $H_{0}:\mu_{X}-\mu_{Y}=-1.0$ vs. $H_{a}:\mu_{X}-\mu_{Y}<-1.0$. Describe, in words, what $H_{a}$ says.

**Solution:**

$H_a$ represents the true mean heart rate recovery rate for non-athletes differs by that of the athlete's by more than 1 beat per minute

### (b) (5 points) 
At level $\alpha=0.01$, find the level-a rejection region $R(\alpha,\theta)$.

**Solution:**

$$
R(\alpha, \theta) = (-\infty,\ \hat{\theta} - q_{\alpha}\cdot SE]
$$

We know:
- $\hat{\theta} = \mu_X - \mu_Y = 0.64 - 2.05 = -1.41$
- $q_\alpha = t_\alpha(df) = 2.82$, $df = min(n-1, m-1) = min(9, 10) = 9$

![[Pasted image 20260307193823.png]]

- $SE = \sqrt{\frac{\hat{\sigma}_X^2}{n} + \frac{\hat{\sigma}_Y^2}{m}} = \sqrt{\frac{0.2}{10} + \frac{0.4}{11}} = \sqrt{0.02 + 0.0364} = \sqrt{0.0564} = 0.237$

Using what we know:
$$
\begin{align*}
R(\alpha, \theta) &= (-\infty,\ \theta_0 - q_{\alpha}\cdot SE]\\
&= (-\infty,\ -1 - 2.82 \cdot 0.237]\\
R(\alpha, \theta) &= (-\infty,\ -1.668]
\end{align*}
$$

### (c) (5 points) 
For the same $\alpha=0.01$ find the lower $(1-\alpha)$ confidence interval $CI_{lower}(\alpha,\theta)$ for $\theta=\mu_{X}-\mu_{Y}$.

**Solution:**

$$
\begin{align*}
CI_{lower}(\alpha, \theta) &= (-\infty,\ \hat{\theta} + q_\alpha\cdot SE]\\
&= (-\infty,\ (0.64 - 2.05) + (2.82)(0.237)]\\
&= (-\infty,\ -1.41 + 0.668]\\
&= (-\infty,\ -0.742]
\end{align*}
$$

### (d) (5 points) 
What is the relationship between the rejection region $R(\alpha,\theta)$ and the lower confidence interval $CI_{lower}(\alpha,\theta)$?

**Solution:**

The $CI_{lower}$ contains all values for the null hypothesis $\theta_0$ that would not be rejected by the test

### (e) (5 points) 
What is the probability of a Type-II error when the actual difference between $\mu_{X}$ and $\mu_{Y}$ is $\mu_{X}-\mu_{Y}=-1.2$?

**Solution:**

$$
\begin{align*}
P(\text{Type-II}) &= P(\hat{\theta} > \theta_0 - q_\alpha\cdot SE | \frac{\hat{\theta} - \theta_a}{SE} \text{ is true})\\
&= P(\frac{\hat{\theta} - \theta_a}{SE} > \frac{\theta_0 - \theta_a}{SE} - q_\alpha)\\
&= P(t(9) > \frac{-1-(-1.2)}{0.237} - 2.82)\\
&= P(t(9) > \frac{0.2}{0.237} - 2.82)\\
&= P(t(9) > -1.976)\\
&= \boxed{0.96}
\end{align*}
$$

### (f) (5 points) 
Find the p-value for a hypothesis test of $H_{0}$ vs. $H_{a}$ and conclude what your decision will be at level $\alpha=0.1$

**Solution:**

$$
\begin{align*}
\hat{T} &= \frac{\hat{\theta} - \theta_0}{SE}\\
&= \frac{(-1.41) - (-1)}{0.237}\\
&= \frac{-0.41}{0.237}\\
&= -1.73
\end{align*}
$$

By using $\hat{T}$, we can find the p-value to be:

$$
\text{p-value} = 0.0588
$$

![[Pasted image 20260307201001.png]]

As such we can conclude to reject the null hypothesis as $\text{p-value} = 0.0588 < 0.1 = \alpha$