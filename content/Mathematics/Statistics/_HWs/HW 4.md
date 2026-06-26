## 1. 

For each of the following questions you only need to write down the final answer with a brief justification for why you claim your answer is true. **You don’t need to derive anything**.

### **(a) (5 points)** Let $X_{1},X_{2},...,X_{n} \overset{iid}{\sim} Ber(p)$. What is the sampling distribution of $n \cdot \overline{X}$?

**SOLUTION**

$$
\begin{align*}
n \cdot \overline{X} &= n \cdot \frac{1}{n}\sum_{i = 1}^{n} X_i\\
&= \sum_{i = 1}^n X_i\\
&\sim \boxed{Bin(n, p)}
\end{align*}
$$

By the definition of the Binomial Distribution
    
### **(b) (5 points)** Let $X_{1},X_{2},...,X_{n} \overset{iid}{\sim} Poi(\lambda)$. What is the sampling distribution of $n \cdot \overline{X}$?

**SOLUTION**

$$
\begin{align*}
n \cdot \overline{X} &= n \cdot \frac{1}{n}\sum_{i = 1}^n X_i\\
&= \sum_{i = 1}^n X_i\\
&\sim \boxed{Poi(n \cdot \lambda)}
\end{align*}
$$

By the Algebra of Poison Random Variables
    
### **(c) (5 points)** Let $X_{1},X_{2},...,X_{n} \overset{iid}{\sim} N(0,1)$. What is the sampling distribution of $\overline{X}$?

**SOLUTION**

$$
\begin{align*}
\overline{X} &= \frac{\sum_{i = 1}^n X_i}{n}\\
&\sim \boxed{N(0, \frac{1}{n})}
\end{align*}
$$

By the Algebra of Normal Random Variables
    
### **(d) (5 points)** Let $X_{1},X_{2},...,X_{n} \overset{iid}{\sim} N(0,1)$. What is the sampling distribution of $\frac{(n-1)}{\sigma^{2}} \times \hat{\sigma}^{2}$?
>_Hint: Refer to Probability, Statistics & Data, Theorem 5.4, or Week-4 slides._

**SOLUTION**

By Theorem 5.4:

$$
\frac{(n-1)}{\sigma^2} \times \hat{\sigma}^2
$$

has a $\chi^2$ distribution with $n-1$ degrees of freedom

Therefore:

$$
\frac{(n-1)}{\sigma^2} \times \hat{\sigma}^2 \sim \boxed{\chi^2(n-1)}
$$
    
### **(e) (10 points)** Let $X \sim N(\mu, \sigma^{2})$ and $Z \sim N(0,1)$. Let $a, b \in \mathbb{R}$. What is the relationship between $a$ and $b$ such that $\mathbb{P}(X \le a) = \mathbb{P}(Z \le b)$? If $\mu=1$, $\sigma=2$, and $b=2.5$, draw an illustration of these two quantities in the same plot.

**SOLUTION**

By the process of **Standardization**:

$$
\frac{X-\mu}{\sigma} \sim N(0, 1) = Z
$$

as such the relationship between $a$ and $b$ is:

$$
\frac{a - \mu}{\sigma} = b
$$

When $\mu = 1$, $\sigma = 2$, and $b = 2.5$, $a$ can be found as:

$$
\begin{align*}
\frac{a - 1}{2} &= 2.5\\
a &= (2.5 \cdot 2) + 1\\
a &= 6
\end{align*}
$$

Below is the plot illustration of the two distributions where 
- Orange → $\mathbb{P}(Z \leq b)$
- Blue → $\mathbb{P}(X \leq a)$

![[Pasted image 20260220172853.png]]

---
## 2.

Bob is a budding social media influencer who is hoping to make it big in the TikTok influencer space. Suppose you work at TikTok, and you know that each of Bob’s TikToks go viral with probability $p = 10\%$.

### **(a) (5 points)** For a TikTok posted by Bob, let $X$ be the outcome where $X=1$ if viral and $X=0$ if not. What is the distribution of $X$?

**SOLUTION**

$$
X \sim \boxed{Ber(p)}
$$
- **(b) (5 points)** Bob conducts an experiment with $n=100$ TikToks. In words, what does $n \cdot \overline{X} = \sum_{i=1}^{n} X_{i}$ measure?

**SOLUTION**

$$
n \cdot \overline{X} = \sum_{i = 1}^n X_i = \text{Number of Viral TikToks out of } 100 \text{ TikToks}
$$
### **(c) (5 points)** Write the sampling distribution of $Y = n \cdot \overline{X}$ and its PMF $p_{Y}(v) = \mathbb{P}(n\overline{X} = v)$.

**SOLUTION**

$$
\begin{align*}
Y &= n \cdot \overline{X} \sim Bin(n, p)\\
\mathbb{P}(Y = v) &= \binom{n}{v}p^v(1-p)^{n-v}
\end{align*}
$$
    
### **(d) (5 points)** Write Bob’s best guess $\hat{p}$ for $p$ in terms of $X_{1}, X_{2}, \dots, X_{n}$.

**SOLUTION**

$$
\hat{p} = \frac{1}{n}(X_1 + X_2 + \cdots + X_n) = \boxed{\frac{1}{n}\sum_{i = 1}^n X_i}
$$
### **(e) (5 points)** In words, describe the events $\{\hat{p} > u\}$ for $u \in (0,1)$ and $\{n \cdot \overline{X} > v\}$ for $v \in (0,100)$.

**SOLUTION**

$\{\hat{p} > u\} =$ Bob's experiment observed his TikTok going viral has a probability greater than some probability $u$

$\{n \cdot \overline{X} > v\} =$ The number of videos going viral observed is more than some value $v$
    
### **(f) (5 points)** If $\{\hat{p} > u\} = \{n \cdot \overline{X} > v\}$, what is the relationship between $u$ and $v$?

**SOLUTION**

From the previous subproblems, we can find the event $\{\hat{p} > u\}$
$$
\begin{align*}
\hat{p} &> u\\
\to \frac{n\overline{X}}{n} &> u\\
\to n\overline{X} &> n \cdot u\\
\end{align*}
$$

Given the condition, we can compare that

$$
\begin{align*}
n\overline{X} &> n \cdot u \\
n\overline{X} &> v
\end{align*}
$$

For these two comparisons to have the same number of occurrence, the following relationship must be true:

$$
\boxed{v = n \cdot u}
$$
    
### **(g) (5 points)** Write the mathematical expression for event $A$: Bob’s estimate $\hat{p}$ is greater than 20%.

**SOLUTION**

$$
A = \{\hat{p} > 20\%\}
$$
    
### **(h) (5 points)** Using parts (e) and (f), write the final expression for the probability that $\hat{p} > 20\%$ in terms of a Binomial probability.

**SOLUTION**

$p = 0.1$
$u = 0.2$
$v = 0.2 \cdot 100 = 20$

$$
\begin{align*}
\mathbb{P}(\hat{p} > 0.2) &= \mathbb{P}(n\overline{X} > 20)\\
&= \sum_{i = 21}^{100} \binom{100}{i}(0.1)^i(0.9)^{100-i}\\
&= \boxed{0.09\%}
\end{align*}
$$

---
## 3. 

Bob recognizes that his answer for Question 2 is annoying to compute since it involves sums of Binomial probabilities. So, he remembers that there was some way of using the central limit theorem to get the final answer.

### **(a) (5 points)** What is $\mathbb{E}(X)$ and $Var(X)$ for $X$ in Question 2?


**SOLUTION**

$$
\begin{align*}
\mathbb{E}(X) &= n \cdot p\\
Var(X) &= n \cdot p (1-p)
\end{align*}
$$

### **(b) (5 points)** Using HW-3 properties, what is $\mathbb{E}(\hat{p})$ and $SD(\hat{p})$?

**SOLUTION**

$$
\begin{align*}
\mathbb{E}(\overline{X}) &= \mu\\
\hat{p} = \overline{X} &\text{, } p = \mu\\
\therefore \mathbb{E}(\hat{p}) &= p = 0.1\\\\

Var(\overline{X}) &= \frac{\sigma^2}{n}\\
\hat{p} = \overline{X} &\text{, } \sigma^2 = p(1-p)\\
Var(\hat{p}) &= \frac{p(1-p)}{n}\\
\therefore SD(\hat{p}) &= \sqrt{Var(\hat{p})} = \sqrt{\frac{p(1-p)}{n}}
\end{align*}
$$


### **(c) (5 points)** Using the CLT, let $S = \frac{\hat{p}-p}{\sqrt{\frac{p(1-p)}{n}}}$. What is the approximate distribution of $S$?

**SOLUTION**

Since $S$ follows the CTL's equation where:

$\mathbb{E}(\hat{p}) = p$
$SD(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}$

$S$ has an approximate distribution of $N(0,1)$
    
### **(d) (5 points)** For $Z \sim N(0,1)$, what is the relationship between $u$ and $w$ such that $\{\hat{p} > u\} \approx \{Z > w\}$?

**SOLUTION**

From the previous subproblems, we can find the event $\{\hat{p} > u\}$
$$
\begin{align*}
\hat{p} &> u\\
\frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}} &> \frac{u - p}{\sqrt{\frac{p(1-p)}{n}}}\\
S &> \frac{u - p}{\sqrt{\frac{p(1-p)}{n}}}
\end{align*}
$$

Given the condition $\{\hat{p} > u\} \approx \{Z > w\}$, we can compare that

$$
\begin{align*}
S &> \frac{u - p}{\sqrt{\frac{p(1-p)}{n}}}\\
Z &> w
\end{align*}
$$

For these two comparisons to have the same number of occurrence, the following relationship must be true:

$$
\boxed{w \approx \frac{u - p}{\sqrt{\frac{p(1-p)}{n}}}}
$$
    
### **(e) (5 points)** Express the probability of event $A$ from Question 2(g) using the standard normal CDF $\Phi$.
    

**SOLUTION**

From 2g: 
$A = \{\hat{p} > 20\%\}$
$p = 0.1$
$u = 0.2$
$n = 100$

$$
\begin{align*}
w &\approx \frac{0.2 - 0.1}{\sqrt{\frac{0.1(0.9)}{100}}}\\
&\approx \frac{0.1 \cdot 10}{\sqrt{0.09}}\\
&\approx \frac{1}{0.3} \\
w &\approx 3.33
\end{align*}
$$

With the calculated $w$:

$$
\mathbb{P}(A) = \mathbb{P}(Z > 3.33) = 1 - \mathbb{P}(Z \leq 3.33) = \boxed{1 - \Phi(3.33)}
$$

---
## 4. 

A survey was conducted on n = 200 participants from the United States asking: “Who did you vote for in the 2024 elections?”. A summary of the responses is below:

| **Response** | **Respondents** |
| ------------ | --------------- |
| Red Party    | 85              |
| Blue Party   | 115             |
### **(a) (5 points)** Let $X$ be a placeholder for whether a participant votes Blue. What is the distribution for $X$?

**SOLUTION**

Since $X$ represents individual trials of participant:

$$
X \sim Ber(p)
$$

### **(b) (5 points)** What is the population parameter of interest? Interpret its meaning.

**SOLUTION**

The population parameter of interest is $p$: the true proportion of all voters in the US who votes Blue
    
### **(c) (5 points)** What is the best guess $\hat{p}$ based on the data? Is this a statistic or a parameter?

**SOLUTION**

$$
\hat{p} = \overline{X} = \frac{\sum_{i = 1}^n X_i}{n} = \frac{115}{200} = \boxed{0.575}
$$

Since this is a value calculated from sample data, this is a **statistic**
    
### **(d) (5 points)** Write the sampling distribution for $\hat{p}$.

**SOLUTION**

$$
\begin{align*}
X_1 \cdots X_n &\overset{iid}{\sim} Ber(p)\\
n \hat{p} &\sim Bin(n, p)\\
\hat{p} &\sim \boxed{\frac{1}{n}Bin(n, p)}
\end{align*}
$$
    
### **(e) (5 points)** Construct a 95% confidence interval for $p$.

**SOLUTION**

![[Pasted image 20260220201210.png]]

Given a $95\%$ interval, the value for $l_\alpha$ and $u_\alpha$ are: $\pm 1.96$
$\hat{p} = 0.575$

Finding Standard Error:

$$
\begin{align*}
SE(\hat{p}) &= \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\\
&= \sqrt{\frac{0.575 \cdot 0.425}{200}}\\
&\approx \boxed{0.0349}
\end{align*}
$$

Now having all the information needed, find the Confidence Interval:

$$
\begin{align*}
CI &= [\hat{p} + l_\alpha SE(\hat{p}), \hat{p} + u_\alpha SE(\hat{p})]\\
&= [0.575 - 1.96(0.0349), 0.575 + 1.96(0.0349)]\\
&= [0.575 - 0.0684, 0.575 + 0.0684]\\
CI&= \boxed{[0.507, 0.643]}
\end{align*}
$$

    
### **(f) (5 points)** Provide a brief interpretation of the interval from part (e).

**SOLUTION**

The Confidence Interval found represents a $95\%$ confident that the true proportion of US voters voting for the Blue party to be between $50.7\%$ and $64.3\%$

### **(g) (5 points)** A critic says this interval is only valid if $\hat{p}$ is Normal or approximately Normal. Is this true for this data? Explain.

**SOLUTION**

The critic is right that this interval is valid only when $\hat{p}$ is approximately Normal since it relies on the assumption that the sampling distribution is a Normal Curve (Bell)

This data follows a Normal Distribution since both Red and Blue party counts are large enough to model using Normal distribution.
    
### **(h) (5 points)** A headline says "The Blue party will win the 2024 elections!". Does this align with your findings? Rephrase it better.

**SOLUTION**

This headline is not accurate as the lower bound found was $50.4\%$, which has a decent probability that the result of the election will result in a tie or Red Party's victory.

A better way to phrase this finding is:

"The Blue party is currently in favor to win, but the results remain close"

---
## 5. 

A survey was conducted on $n = 200$ participants from UCSD asking: “Do you think Artificial Intelligence (AI) is going to replace our jobs?”. In addition to the responses to this question, the participants were also asked what their broad major was. The breakdown of the responses by major is below:

| **Major** \ **Response** | **Yes** | **No** |
| ------------------------ | ------- | ------ |
| Science & Engineering    | 80      | 40     |
| Arts & Humanities        | 50      | 30     |

We are interested in constructing a confidence interval for the difference in opinions about AI for Science & Engineering majors vis-á-vis Arts & Humanities majors. Let $X$ be the response of a randomly chosen student with a Science & Engineering major, and let $Y$ be the response of a randomly chosen Arts & Humanities major with population parameters $p_X$ and $p_Y$ respectively

### **(a) (5 points)** What is an appropriate distribution for $X$ and $Y$ ?

**SOLUTION**

$$
\begin{align*}
&X \sim Ber(p_X) &Y \sim Ber(p_Y) 
\end{align*}
$$

### **(b) (5 points)** Let $\theta = p_{X} - p_{Y}$. Interpret this in context.

**SOLUTION**

$p_X$: the true proportion of the Science and Engineering students answering "Yes"
$p_Y$: the true proportion of the Arts and Humanities students answering "Yes"

As such, the expression $\theta$ represents the true difference between the proportion of the 2 categories of majors believe that AI is going to replace their jobs

### **(c) (5 points)** Write the expression for $\hat{\theta}$ and compute its value.

**SOLUTION**

$$
\begin{align*}
\hat{p}_X &= \frac{80}{120} = 0.667\\
\hat{p}_Y &= \frac{50}{80} = 0.625\\
\\
\hat{\theta} &= \hat{p}_X - \hat{p}_Y \\
&= 0.667 - 0.625\\
&= \boxed{0.042}
\end{align*}
$$   
### **(d) (5 points)** Compute $SE(\hat{\theta})$, the standard error for the estimator.

**SOLUTION**

$$
\begin{align*}
SE(\hat{\theta}) &= \sqrt{\frac{\hat{p}_X(1-\hat{p}_X)}{n_X} + \frac{\hat{p}_Y(1-\hat{p}_Y)}{n_Y}}\\
&= \sqrt{\frac{0.667 \times 0.333}{120} + \frac{0.625 \times 0.375}{80}}\\
&= \sqrt{\frac{0.2221}{120} + \frac{0.2344}{80}}\\
&= \sqrt{0.00185 + 0.00293}\\
&= \sqrt{0.00478}\\
&= \boxed{0.0691}
\end{align*}
$$
    
### **(e) (5 points)** Write the expression for a $100 \times (1-\alpha)\%$ confidence interval for $\hat{\theta}$ in terms of $z_{\alpha/2}$.

**SOLUTION**

$$
\begin{align*}
CI(\alpha, p) &= \hat{\theta} \pm z_{\alpha/2} \times SE(\hat{\theta})\\
&= \boxed{(\hat{p}_X - \hat{p}_Y) \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}_X(1-\hat{p}_X)}{n_X} + \frac{\hat{p}_Y(1-\hat{p}_Y)}{n_Y}}}
\end{align*}
$$
    
### **(f) (5 points)** Compute $z_{\alpha/2}$ for $\alpha=0.1$ using a web applet (include screenshot).

**SOLUTION**

![[Pasted image 20260220230207.png]]

When $\alpha = 1 \to 1-\alpha = 0.9$

Based on the web app, the value for $z_{\alpha/2}$ is given $\boxed{\pm1.64}$ 
    
### **(g) (5 points)** Write the final 90% confidence interval for $\theta$.

**SOLUTION**

$$
\begin{align*}
CI(\alpha, \theta) =
&\Bigg[(\hat{p}_X - \hat{p}_Y) - z_{\alpha/2} \times \sqrt{\frac{\hat{p}_X(1-\hat{p}_X)}{n_X} + \frac{\hat{p}_Y(1-\hat{p}_Y)}{n_Y}}\\
&, (\hat{p}_X - \hat{p}_Y) + z_{\alpha/2} \times \sqrt{\frac{\hat{p}_X(1-\hat{p}_X)}{n_X} + \frac{\hat{p}_Y(1-\hat{p}_Y)}{n_Y}}\Bigg]\\
= &\Bigg[(0.667 - 0.625) - 1.64 \times 0.0691, (0.667 - 0.625) + 1.64 \times 0.0691\Bigg]\\
= &\Big[0.042 - 0.1137, 0.042 + 0.1137\Big]\\
= &\boxed{[-0.0717, 0.1557]}
\end{align*}
$$
    
### **(h) (5 points)** Is there sufficient evidence of a difference in perceptions between the groups? Why/why not?

**SOLUTION**

There are no evidence of a difference in perceptions between the two groups because with the difference's confidence interval including $0$ suggests there is no significant difference between the result of the two groups.

---
## 6. 

You work at a reputable polling agency covering the elections, and have recently conducted a
survey with $n = 100$ participants. You collect their responses $X_1, X_2, \cdots , X_n \overset{iid}{\sim}Ber(p)$ and compute $\hat{p}$. Based on your analysis, you find that the margin of error at the $90\%$ confidence level is $0.082$. You report these results to your boss, who says that the margin of error is too high, and that the margin of error should be less than $0.05$ at the $99\%$ confidence level. Assuming your estimate of $\hat{p}$ remains the same as your original survey, what is the minimum number of participants you need in your new survey to satisfy your boss’s requirements?
Provide screenshots justifying the use of any quantiles you use in this problem.

**SOLUTION**

![[Pasted image 20260220230207.png]]

With a $90\%$ confidence, the $z_{\alpha/2} = 1.64$

Find the current $\hat{p}$

$$
\begin{align*}
ME &= z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\\
0.082 &= 1.64 \times \sqrt{\frac{\hat{p}(1-p)}{100}}\\
0.0498 &= \sqrt{\frac{\hat{p}(1-\hat{p})}{100}}\\
\frac{\hat{p}(1-\hat{p})}{100} &= 0.00248\\
\hat{p}(1-\hat{p}) &= 0.248
\end{align*}
$$

With this information, we can find the $ME < 0.05$ given a $99\%$ confidence interval to find $n$:

![[Pasted image 20260220232627.png]]

$z_{\alpha/2} = 2.58$ when at $99\%$ confidence

$$
\begin{align*}
ME' = z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} &< 0.05\\
2.58 \times \sqrt{\frac{0.248}{n}} &< 0.05\\
\sqrt{\frac{0.248}{n}} &< 0.0194\\
\frac{0.248}{n} &< 0.0003767\\
\frac{n}{0.248} &> \frac{1}{0.0003767}\\
n &> \frac{0.248}{0.0003767}\\
n &> 658.35 \\
n &\to \boxed{659}
\end{align*}
$$

---
## 7. 

An $85\%$ confidence interval for a population mean, $\mu$, is given as $(18.985, 21.015)$. This
confidence interval is based on a simple random sample of $n = 10$ observations. Calculate the sample mean $X$ and standard deviation $\hat{\sigma}$ which reproduce this confidence interval. Assume that all conditions necessary for inference are satisfied, and use the student’s t distribution wherever needed. Provide a screenshot justifying the use of any quantiles you use in this problem.

**SOLUTION**

$n = 10$
$df = 9$

Finding $\overline{X}$
$$
\overline{X} = \frac{l_\alpha + u_\alpha}{2} = \frac{18.985 + 21.015}{2} = \frac{40}{2} = \boxed{20}
$$

Finding $s$

![[Pasted image 20260220234846.png]]

$t_{\alpha/2, df=9} = 1.57$ at $85\%$ confidence

For a t-distribution, the Margin of Error is: 
$$
\begin{align*}
ME &= t_{\alpha/2, df=9} \times \frac{s}{\sqrt{n}}\\
21.015 - 20 &= 1.57 \times \frac{s}{\sqrt{10}}\\
1.015 &= 1.57 \times\frac{s}{3.162}\\
s &= \boxed{2.04}
\end{align*}
$$

---
## 8. 

The table below summarizes an experiment to answer this question. A total of 20 participants were recruited for the experiments and randomly assigned to two groups of size $n_X = 12$ and $n_Y = 8$. In the first group ($X$), the participants were required to not use any electronic devices for at least an hour before they go to sleep. In the second group ($Y$), the participants were asked to doomscroll their favorite social-media platform before falling asleep. A wearable health device collected their sleep quality scores, and the sleep quality data is summarized in the table below:

| **Group** | **n** | **$\hat{\mu}$** | **$\hat{\sigma}$** |
| --------- | ----- | --------------- | ------------------ |
| X         | 12    | 70.0            | 5.0                |
| Y         | 8     | 60.0            | 10.0               |
### **(a) (5 points)** Let $\theta = \mu_{X} - \mu_{Y}$. Interpret this in context.

**SOLUTION**

$\mu_X$ represents the true mean of group $X$'s sleep quality
$\mu_Y$ represents the true mean of group $Y$'s sleep quality

As such, the expression $\theta$ represents the true difference in the mean of sleep quality scores between group $X$ and $Y$

### **(b) (5 points)** What does 70.0 represent? Is it a parameter or statistic?

**SOLUTION**

The value $70.0$ represents the sample mean of group $X$'s sleep quality score

Since this value is from the observation of the sample space, this is a **statistic**
    
### **(c) (5 points)** Write the expression for $\hat{\theta}$ and compute it.

**SOLUTION**

$$
\begin{align*}
\hat{\theta} &= \hat{\mu}_X - \hat{\mu}_Y\\
&= 70 - 60\\
\hat{\theta} &= \boxed{10}
\end{align*}
$$
### **(d) (5 points)** Write the expression for the $(1-\alpha)$ confidence interval for $\theta$ in terms of $\hat{\mu}$, $\hat{\sigma}$, and $t_{\alpha/2}(d)$. Compute the degrees of freedom $d$.

**SOLUTION**

Confidence Interval:

$$
(\hat{\mu}_X - \hat{\mu}_Y) \pm t_{\alpha/2}(d)\times\sqrt{\frac{\hat{\sigma}_X^2}{n_X} + \frac{\hat{\sigma}_Y^2}{n_Y}}
$$

Degree of Freedom:

$$
\begin{align*}
df &= min\{n_X-1, n_Y - 1\}\\
df &= min\{11, 7\}\\
df &= \boxed{7}
\end{align*}
$$

### **(e) (5 points)** If $\alpha=0.05$, what is $t_{\alpha/2}$? Include a screenshot.

**SOLUTION**

![[Pasted image 20260221001201.png]]

Based on this, $t_{\alpha/2} = 2.36$ when $\alpha = 0.05$
    
### **(f) (5 points)** Compute the 95% confidence interval for $\theta$.

**SOLUTION**

$$
\begin{align*}
CI(\alpha, \theta) &= \Bigg[(\hat{\mu}_X - \hat{\mu}_Y) - t_{\alpha/2}(d)\times\sqrt{\frac{\hat{\sigma}_X^2}{n_X} + \frac{\hat{\sigma}_Y^2}{n_Y}}, (\hat{\mu}_X - \hat{\mu}_Y) + t_{\alpha/2}(d)\times\sqrt{\frac{\hat{\sigma}_X^2}{n_X} + \frac{\hat{\sigma}_Y^2}{n_Y}}\Bigg]\\
&= \bigg[10 - 2.36 \times \sqrt{\frac{5^2}{12} + \frac{10^2}{8}}, 10 + 2.36 \times\sqrt{\frac{5^2}{12} + \frac{10^2}{8}}\bigg]\\
&= \Big[10 - 2.36 \times \sqrt{2.083 + 12.5}, 10-2.36 \times \sqrt{2.083 + 12.5}\Big]\\
&= \Big[10 - 2.36 \times \sqrt{14.583}, 10 + 2.36 \times \sqrt{14.583}\Big]\\
&= \big[10 - 2.36 \times 3.819, 10 + 2.36 \times 3.819\big]\\
&= \big[10 - 9.012, 10 + 9.012\big]\\
&= \boxed{[0.988, 19.012]}
\end{align*}
$$
    
### **(g) (5 points)** Interpret the interval. Is there sufficient evidence that doomscrolling impacts sleep scores?

**SOLUTION**

With $95\%$ confidence, the true difference in mean sleep quality scores between those who avoid electronics and those who doomscroll falls between $0.987$ and $19.013$ points

There is sufficient evidence that doomscrolling impacts sleep scores since the confidence interval did not pass through $0$, suggesting a significant improvement of sleep quality when avoiding electronics before sleep.