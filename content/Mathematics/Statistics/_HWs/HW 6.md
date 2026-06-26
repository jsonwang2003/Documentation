## Question 1
A sample of $n=300$ urban adult residents of CA revealed that 120 favorably approved of the incumbent president's job performance, whereas a sample of $m=180$ rural residents yielded 75 who favorably approved of the incumbent president. We are interested in testing whether or not there is a difference in perception of the incumbent president's performance across the two groups.

### (a) (5 points) Let $X_1, X_2, \dots, X_n$ be the responses of the urban residents and $Y_1, Y_2, \dots, Y_m$ be the responses of the rural residents. In the setting of this problem, describe the distributions these random variables are sampled from.

**SOLUTION**

$$
\begin{align*}
X_1, X_2, ..., X_{300} &\overset{iid}{\sim} Ber(p_X)\\
Y_1, Y_2, ..., Y_{180} &\overset{iid}{\sim} Ber(p_Y)
\end{align*}
$$

### (b) (5 points) Identify the main parameter of interest, $\theta$.

**SOLUTION**

$$
\theta = p_X - p_Y
$$

### (c) (5 points) Write down the expression for the statistic $\hat{\theta}$, which is our best guess for the population parameter $\theta$.

**SOLUTION**

$$
\hat{\theta} = \hat{p}_X - \hat{p}_Y
$$
### (d) (5 points) Write down the null and alternative hypothesis for the question.

**SOLUTION**

- $H_0$ : $\theta = \theta_0 = 0$
- $H_a$ : $\theta \ne 0$

### (e) (5 points) What would the ideal rejection region look like for rejecting $H_0$ in favor of $H_a$?

**SOLUTION**

The ideal rejection region will be the 2 sides away from $\theta_0$. We reject $H_0$ if our observed statistic $\hat{\theta}$ is significantly larger than 0 or significantly smaller than 0.

### (f) (5 points) Assuming the null hypothesis $H_0$ is true, what is the sampling distribution of $\hat{\theta}$? (Hint: Use CLT approximation for the sampling distribution of the statistic $\hat{\theta}$ which we have encountered in class earlier. You just need to write down what this sampling distribution is under $H_0$)

**SOLUTION**

$$
\frac{\hat{\theta} - \theta_0}{SE} \approx N(0,1) \equiv Z
$$

### (g) (15 points) By setting $\alpha$ to be the Type-I error probability, write down the final expression for the rejection region $R(\alpha, \theta)$ in terms of $z_{\alpha/2}$.

**SOLUTION**

$$
\begin{align*}
R(\alpha, \theta) &= (-\infty,\ \theta_0 - z_{\alpha/2} \times SE] \ \cup \ [\theta_0 + z_{\alpha/2} \times SE,\ \infty)\\\\

\hat{p}_{pooled} &= \frac{n\hat{p}_X + m\hat{p}_Y}{n+m}\\
&= \frac{120 + 75}{ 300 + 180}\\
&= \frac{195}{480}\\
&= 0.406\\\\

SE &= \sqrt{\hat{p}_{pooled} \times (1 - \hat{p}_{pooled}) \times \{\frac{1}{n} + \frac{1}{m}\}}\\
&= \sqrt{(0.406) \times (0.594) \times \{\frac{1}{300} + \frac{1}{180}\}}\\
&= \sqrt{0.00214}\\
&= 0.0463\\\\

R(\alpha, \theta) &= (-\infty,\ 0 - z_{\alpha/2} \times (0.09428)] \ \cup \ [0 + z_{\alpha/2} \times (0.09428),\ \infty)\\
&= (-\infty,\ 0.0463 \cdot -z_{\alpha/2}] \ \cup \ [0.0463 \cdot z_{\alpha/2},\ \infty)
\end{align*}
$$

### (h) (5 points) Fixing $\alpha=0.01$ find the level-$\alpha$ rejection region $R(\alpha, \theta)$.

**SOLUTION**

![[Pasted image 20260314173127.png]]
$z_{\alpha/2} = 2.58$

$$
\begin{align*}
R(\alpha, \theta) &= (-\infty,\ 0.0463 \cdot -z_{\alpha/2}] \ \cup \ [0.0463 \cdot z_{\alpha/2},\ \infty)\\
&= (-\infty,\ 0.0463 \cdot -2.58] \ \cup \ [0.0463 \cdot 2.58,\ \infty)\\
&= (-\infty,\ -0.1194] \ \cup \ [0.1194,\ \infty)
\end{align*}
$$
### (i) (5 points) What is your final decision is based on the level $\alpha=0.01$ hypothesis test?

**SOLUTION**

$\hat{\theta} = \hat{p}_X - \hat{p}_Y = \frac{120}{300} - \frac{75}{180} = -0.0166$

Since $\hat{\theta}$ is not within the rejection region, we failed to reject $H_0$ with $\alpha = 0.01$

### (j) (5 points) Compute the p-value for the hypothesis test, and specify what your decision will be if you were to, instead, perform a level $\alpha=0.05$ hypothesis test.

**SOLUTION**

$$
\begin{align*}
\hat{T}_{obs} &= \frac{\hat{\theta} - \theta_0}{SE}\\
&= \frac{(-0.0166) - 0}{0.0463}\\
&= -0.3585\\\\
\text{p-value} &= 2 \times z_{\alpha/2}\\
&= 2 \times 0.36\\
&= 0.72
\end{align*}
$$

Since p-value $= 0.72 > 0.05 = \alpha$ we have failed to reject the null hypothesis with $\alpha = 0.05$

---
## Question 2
In a study to estimate the average height of adult male basketball players, a researcher wants to test if the average height is greater than 200cm. Prior studies indicate that the variance in height is $16\text{cm}^2$.

### (a) (10 points) Write down any assumptions about the data and identify the setting of the problem.

**SOLUTION**

$X$: The average height of a random chosen male adult basketball player

$$
X_1,X_2, ..., X_n \overset{iid}{\sim} N(\mu, \sigma^2)
$$

### (b) (10 points) From part (a), identify the relevant population parameter, $\theta$, and the sample statistic, $\hat{\theta}$, the researcher will use to make any statistical inference.

**SOLUTION**

$\theta$: the true population mean of the height of adult male basketball players ($\mu$)
$\hat{\theta}$: The sample proportion mean of the height of adult male basketball players ($\overline{X}$)

### (c) (10 points) The researcher wants to compute a two-sided 99% confidence interval for the sample statistic $\theta$. If they want the margin of error to be 0.01cm, what is the minimum number of samples needed?

**SOLUTION**

- $\alpha = 0.01$
- $ME(\alpha, \theta) = 0.01$
- $SE = \frac{\sigma}{\sqrt{n}}$
- $z_{\alpha/2} = 2.58$

![[Pasted image 20260314204024.png]]

$$
\begin{align*}
ME(\alpha, \theta) &= q_{\alpha/2}\times SE\\
&= z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}\\
n &= \frac{z_{\alpha/2} \times \sigma}{ME}\\
n &= \frac{2.58 \times 4}{0.01}\\
n &= \boxed{1032}
\end{align*}
$$

### (d) (10 points) In part (c), the researcher uses a two-sided confidence interval. In words, describe why/why not this type of a confidence interval is appropriate for the research question they wish to investigate.

**SOLUTION**

The two-sided confidence interval is inappropriate because the research question is directional (testing if heights are greater than 200cm). A two-sided interval wastes statistical power by accounting for a 'less than' direction that the researcher is not investigating; a one-sided lower bound would more accurately align with the alternate hypothesis.

### (e) (10 points) Write down the appropriate null and alternate hypotheses for the question.

**SOLUTION**

$\theta = \mu$

$H_0$ : $\theta = \theta_0 = 200$
$H_a$ : $\theta > 200$

### (f) (10 points) The researcher aims to have a power of 80% to detect an actual average height of 202cm. What sample size is required for this test at a $\alpha=0.01$ significance level?

**SOLUTION**

- $power = 0.8$
- $\mu_a = 202$
- $\alpha = 0.01$

Sampling Distribution: 

$$
\frac{\hat{\theta} - \theta}{SE} \sim N(0,1)\equiv Z
$$

Reject $H_0$ if $\hat{\theta} > \theta_0 + z_{\alpha}\times SE$:

$$
SE = \frac{\sigma}{\sqrt{n}}
$$
Calculate Power:
$$
\begin{align*}
power &= \mathbb{P}(\text{reject } H_0 \ | \ H_a \text{ is true})\\
&= \mathbb{P}(\hat{\theta} > \theta_0 + z_{\alpha} \times SE | \frac{\hat{\theta} - \theta_a}{SE} \sim Z)\\
&= \mathbb{P}(\frac{\hat{\theta} - \theta_a}{SE} > \frac{\theta_0 + z_{\alpha} \times SE - \theta_a}{SE})\\
&= \mathbb{P}(Z > \underbrace{\frac{\theta_0 - \theta_a}{SE} + z_{\alpha}}_{x = z_{1-power}})
\end{align*}
$$
We found the $x = z_{1-power}$ and use it to find $n$
- $x = z_{1-power} = z_{1-0.8} = z_{0.2} = 0.84$

![[Pasted image 20260314211719.png]]

- $z_\alpha = z_{0.01} = -2.33$

![[Pasted image 20260314212906.png]]

$$
\begin{align*}
Z_{1-power} &= \frac{\theta_0 - \theta_a}{SE} + z_{\alpha}\\
&= \frac{\theta_0 - \theta_a}{\frac{\sigma}{\sqrt{n}}} + z_{\alpha}\\\\
n &= ((z_{1-power} - z_\alpha)\times \frac{\sigma}{\theta_0 - \theta_a})^2\\
&= ((0.84 - (-2.33)) \times \frac{4}{200-202})^2\\
&= ((3.17) \times -2)^2\\
&= 6.34^2\\
&= 40.1956\\
n &\approx \boxed{41}
\end{align*}
$$

---

## Question 3
A clinical trial is needed to compare the efficacy of a new diabetes drug $X$ in comparison to the baseline $Y$. Prior pilot studies found the standard deviations for both drugs to be $\sigma_X = 10.0$ units and $\sigma_Y = 12.0$ units. The FDA requires there to be a reduction of $5\mu g/ml$ in blood sugar to be considered "innovation" in order to release the drug into the market. Furthermore, all results need to be reported at a statistical significance level of $\alpha=0.01$.

### (a) (10 points) State the main assumptions in this problem and identify the problem setting.

**SOLUTION**

$X$: The reduced blood sugar of a randomly chosen participant who used the drug $X$
$Y$: The reduced blood sugar of a randomly chosen participant who used the drug $Y$

$$
\begin{align*}
X_1, X_2, ..., X_{n_X} &\overset{iid}{\sim} N(\mu_X, \sigma_X^2)\\
Y_1, Y_2, ..., Y_{n_Y} &\overset{iid}{\sim} N(\mu_Y, \sigma_Y^2)
\end{align*}
$$


### (b) (10 points) Identify the population parameter $\theta$ and sample statistic $\hat{\theta}$ the researchers are interested in.

**SOLUTION**

$\theta$ : The difference of the true means of the new drug $X$ with the baseline $Y$ ($\mu_X - \mu_Y$)
$\hat{\theta}$ : The difference of the sample means of the new drug $X$ with the baseline $Y$ ($\hat{\mu}_X - \hat{\mu}_Y$)

### (c) (10 points) Identify the null and alternate hypotheses for this problem which will enable the researchers to make the necessary statistical inference.

**SOLUTION**

$H_0 : \theta = \theta_0 = 0$
$H_a : \theta > 0$

### (d) (10 points) The units for the standard deviation are intentionally left as units. What units should these be for this problem to make sense?

**SOLUTION**

When calculating sample distribution, the values of the numerator and denominator must be the same for the distribution to be unitless:

$$
\frac{\hat{\theta} - \theta}{SE} \sim N(0,1)
$$
Since we know $\theta_a = 5\ \mu g/ml$, the unit for $SE \to \sigma$ must also have the unit: $\mu g/ml$ 

### (e) (10 points) Suppose the researchers choose to recruit $n$ volunteers for the research study and randomly split half of them to the two groups, i.e., $n/2$ to take drug X and the remaining $n/2$ volunteers to take the drug Y. What is the minimum sample size, $n$, needed to detect if the new drug improves on the baseline with power 90%?

**SOLUTION**

- $power = 0.9$
- $\alpha = 0.1$
- $\sigma_X = 10$, $\sigma_Y = 12$

I. Sampling Distribution:

$$
\frac{\hat{\theta} - \theta}{SE} \sim N(0,1) \equiv Z
$$

where: 

$$
SE = \sqrt{Var(\hat{\theta}}) = \sqrt{\frac{\sigma_X^2}{n/2} + \frac{\sigma_Y^2}{n/2}} = \sqrt{\frac{\sigma_X^2 + \sigma_Y^2}{n/2}}
$$

II. Sampling Distribution if $H_a$ is true:

$$
\frac{\hat{\theta} - \theta_a}{SE} \sim N(0,1)
$$

III. Reject $H_0$ if:

$$
\hat{\theta} > \theta_0 + z_{\alpha} \times SE
$$

IV. power

$$
\begin{align*}
power &= P(\hat{\theta} > \theta_0 + z_{\alpha} \times SE \ | \ \frac{\hat{\theta} - \theta_a}{SE} \sim N(0,1))\\
&= P(\frac{\hat{\theta} - \theta_a}{SE} > \frac{\theta_0 + z_{\alpha} \times SE - \theta_a}{SE})\\
&= P(Z > \underbrace{\frac{\theta_0 - \theta_a}{SE} + z_\alpha}_{x = Z_{power}})
\end{align*}
$$

We found the $x = Z_{power}$ and use it to find $n$
- $x = z_{power} = z_{0.9} = -1.28$

![[Pasted image 20260314224325.png]]

- $z_\alpha = z_{0.01} = 2.33$

![[Pasted image 20260314231511.png]]

$$
\begin{align*}
Z_{power} &= \frac{\theta_0 - \theta_a}{SE} + z_\alpha\\
&= \frac{\theta_0 - \theta_a}{\sqrt{\frac{\sigma_X^2 + \sigma_Y^2}{n/2}}} + z_\alpha\\
Z_{power} - z_\alpha &= \frac{\theta_0 - \theta_a}{\frac{\sqrt{\sigma_X^2 + \sigma_Y^2}}{\sqrt{n/2}}}\\
\sqrt{n/2} &= (Z_{power} - z_\alpha) \times \frac{\sqrt{\sigma_X^2 + \sigma_Y^2}}{\theta_0 - \theta_a}\\
n &= 2 \cdot ((Z_{power} - z_\alpha) \times \frac{\sqrt{\sigma_X^2 + \sigma_Y^2}}{\theta_0 - \theta_a})^2\\
&= 2 \cdot ((-1.28 - (2.33)) \times \frac{\sqrt{10^2 + 12^2}}{0-5})^2\\
&= 2 \cdot ((3.61) \times \frac{15.62}{-5})^2\\
&= 2 \cdot (-11.27764)^2\\
&= 254.37\\
n &\approx \boxed{256}
\end{align*}
$$

---
## Question 4
In an upcoming national election, you are in charge of conducting exit polls to predict the winner. The race is between two parties: the orange party and the purple party. You decide to conduct a one-sided population proportion hypothesis test to assess the proportion of voters favoring the purple party candidate.

### (a) (10 points) Write down the appropriate assumptions about the data, identify the population parameter of interest, $\theta$, and the sample statistic, $\hat{\theta}$, you intend to use.

**SOLUTION**

$X$: The response of a randomly chosen individual whether or not they favor the purple party over the orange party

$$
X_1, X_2, ..., X_n \overset{iid}{\sim} Ber(p)
$$

$\theta$ : The true population proportion that favors the purple party over the orange party ($p$)
$\hat{\theta}$ : The sample proportion  that favors the purple party over the orange party ($\hat{p}$)

### (b) (20 points) Identify the null and the alternate hypotheses which will enable you to make the necessary inference for this question. Describe the sampling distribution of the sample statistic $\hat{\theta}$ under the null hypothesis and the alternate hypothesis.

**SOLUTION**

$H_0 \ : \ \theta = \theta_0 = 0.5$
$H_a : \theta > 0.5$ 

Sampling Distribution if $H_0$ is true:

$$
\frac{\hat{\theta} - \theta_0}{SE_0} \approx N(0, 1) \equiv Z
$$
where 

$$
SE_0 = \sqrt{\frac{p_0(1-p_0)}{n}} 
$$

Sampling Distribution if $H_a$ is true:

$$
\frac{\hat{\theta} - \theta_a}{SE_a} \approx N(0,1) \equiv Z
$$

Standard Error:

$$
SE_a = \sqrt{\frac{p_a(1-p_a)}{n}}
$$
### (c) (20 points) Based on prior studies in electoral contexts, an election is considered to have a "moderate level of support" when the true population proportion is 55% or greater. Anything less than that is considered to be a "small margin". You want your test to have a power of at least 95%, when the true political sentiment in favor of the purple party candidate is a moderate level of support. What is the minimum sample size you would need to achieve this? Assume a significance level of $\alpha=0.01$.

**SOLUTION**

- $\theta_a = p_a = 0.55$
- $\alpha = 0.01$
- $power = 0.95$

Reject $H_0$ if :

$$
\hat{\theta} > \theta_0 + z_\alpha \times SE_0
$$

Power:
$$
\begin{align*}
power &= P(\hat{\theta} > \theta_0 + z_\alpha \times SE_0 \ | \ \frac{\hat{\theta} - \theta_a}{SE_a} \approx Z)\\
&= P(\frac{\hat{\theta} - \theta_a}{SE_a} > \frac{\theta_0 + z_\alpha \times SE_0 - \theta_a}{SE_a})\\
&= P(Z > \underbrace{\frac{\theta_0 - \theta_a}{SE_a} + z_\alpha\frac{SE_0}{SE_a}}_{x = Z_{power}})\\
\end{align*}
$$

We found the $x = Z_{power}$ and use it to find $n$
- $x = z_{power} = z_{0.95} = -1.64$

![[Pasted image 20260314234853.png]]

- $z_\alpha = z_{0.01} = 2.33$

![[Pasted image 20260314231511.png]]

$$
\begin{align*}
Z_{power} &= \frac{\theta_0 - \theta_a}{SE_a} + z_\alpha \cdot \frac{SE_0}{SE_a}\\
&= \frac{\theta_0 - \theta_a}{\sqrt{\frac{p_a(1-p_a)}{n}}} + z_\alpha \cdot \frac{\sqrt{\frac{p_0(1-p_0)}{n}}}{\sqrt{\frac{p_a(1-p_a)}{n}}}\\
&= \frac{\theta_0 - \theta_a}{\frac{\sqrt{p_a(1-p_a)}}{\sqrt{n}}} + z_\alpha \cdot \frac{\frac{\sqrt{p_0(1-p_0)}}{\sqrt{n}}}{\frac{\sqrt{p_a(1-p_a)}}{\sqrt{n}}}\\
&= \frac{\sqrt{n}(\theta_0 - \theta_a)}{\sqrt{p_a(1-p_a)}} + z_\alpha \cdot \frac{\sqrt{p_0(1-p_0)}}{\sqrt{p_a(1-p_a)}}\\
\frac{\sqrt{n}(\theta_0 - \theta_a)}{\sqrt{p_a(1-p_a)}} &= Z_{power} - z_\alpha \cdot \frac{\sqrt{p_0(1-p_0)}}{\sqrt{p_a(1-p_a)}}\\
\sqrt{n} &= (\frac{\sqrt{p_a(1-p_a)}}{\theta_0 - \theta_a}) \times (Z_{power} - z_\alpha \cdot \frac{\sqrt{p_0(1-p_0)}}{\sqrt{p_a(1-p_a)}})\\
n &= [(\frac{\sqrt{p_a(1-p_a)}}{\theta_0 - \theta_a}) \times (Z_{power} - z_\alpha \cdot \frac{\sqrt{p_0(1-p_0)}}{\sqrt{p_a(1-p_a)}})]^2\\
&= [(\frac{\sqrt{0.55 \times 0.45}}{0.5 - 0.55}) \times (-1.64 - 2.33 \cdot \frac{\sqrt{0.5 \times 0.5}}{\sqrt{0.55 \times 0.45}})]^2\\
&= [(\frac{\sqrt{0.2475}}{-0.05}) \times (-1.64 - 2.33 \cdot \frac{\sqrt{0.25}}{\sqrt{0.2475}})]^2\\
&= [(\frac{0.497}{-0.05}) \times (-1.64 - 2.33 \cdot \frac{0.5}{0.497})]^2\\
&= [(-9.94) \times (-1.64 - 2.33 \cdot 1.006)]^2\\
&= [(-9.94) \times (-1.64 - 2.344)]^2\\
&= [(-9.94) \times (-3.984)]^2\\
&= 39.601^2\\
&= 1568.23\\
n &\approx \boxed{1569}
\end{align*}
$$
