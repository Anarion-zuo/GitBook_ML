# Ensemble Learning

## Errors

### Random Error

In regular cases, we assume the randomly generalized error of the model is normal distributed of Laplace distributed, corresponding to L2 and L1.
$$
y=f(x)+\epsilon,\epsilon\sim N(0,\sigma^2)
$$
Such an error is in an extend inevitable, and can only be minimized by larger amount of samples.

### Bias

Bias is defined as the difference between the expectation of the model’s output and the real world.
$$
bias(\hat f)=E[\hat f(x)]-f(x)
$$
where expectation is approximately estimated by the average value of the samples.
$$
E(X)=\frac{1}{N}\sum_{i=0}^NX_i
$$
Separate the training set into many parts. Train the model upon each part and test the model upon the testing set. The bias is measured by the different outputs of the different models.

### Variance

Similar to the expectation, we define the variance of the model to be:
$$
var(\hat f)=E(\hat f(x)-E(\hat f(x))^2)
$$
Variance can be factorized into certain parts.
$$
\begin{align*}
E(\hat f(x)-E(\hat f(x))^2)&=E((f(x)+\epsilon-\hat f(x))^2)\\
&=E((f(x)-\hat f(x))^2)+\sigma^2_\epsilon\\
&=E((f(x)-\bar f(x)+\bar f(x)-\hat f(x))^2)+\sigma^2_\epsilon\\
&=(f(x)-\bar f(x))^2+E((\bar f(x)-\hat f(x))^2)+\sigma_\epsilon^2\\
&=bias^2+variance+random\_error
\end{align*}
,\bar f(x)=E(\hat f(x))
$$
We have to find balance for bias and variance in our model. The fault of under and over fitting, which makes it seem that variance and bias cannot be in harmony together, must be avoided with some effort.

![1554574272057](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1554574272057.png)

In the case of single model, the way of avoiding under and over fitting is as following.

- Under fitting
  - Iterate more
  - Add more features
  - Abandon regular
- Over fitting
  - Larger training set
  - Reduce features
  - Add regular

Variance and bias can both be reduced at the same time with multiple models.

## Bootstrap Aggregating (Bagging)

For a set $D=\{x_1,x_2,...,x_N\}$, randomly pick a sample from the set and do it for multiple times. Thus forms a new set. Suppose the probability of the selection of each sample to be uniformly distributed, $\frac{1}{N}$, since the samples are allowed to be repeatedly selected. The total proportion of the set to be picked is:
$$
\lim_{N\rightarrow\infty}(1-\frac{1}{N})^N=\frac{1}{e}
$$
Repeat the process of bootstrap for $M$ times and get $M$ different training sets. Train and generate $M$ different models with these sets. The output of the final model is the average value of the outputs of all models.
$$
f_{avg}(x)=\frac{1}{M}\sum_{m=1}^Mf_m(x)
$$
Such is the process of aggregating. With more models joining the process of prediction, we can reduce the variance of the final model, for $var(\bar X)=\frac{\sigma^2_X}{M}$. The more models, the less variance. However, since there may be duplicated samples among different training sets, the models are not absolutely irrelevant.
$$
f_{avg}=\rho\sigma^2+(1-\rho)\frac{\sigma^2}{M}
$$

## AdaBoost

### Weighting and Re-weighting

In order to be better than simple Bagging models, we alter the relation between the sub-models of ensemble learning to that the next model only learns upon the samples on which the previous model failed. The implementation is re-weighting the samples for each model. The target function becomes:
$$
J(f)=\sum_{i=1}^Nw_iL(y_i,f(x_i))+\lambda R(f)
$$
Define error/rate of correctness of a specified model m:
$$
\epsilon_m=\frac{\sum_{i=1}^Nw_{m,i}I(y_i\ne\phi_m(x_i))}{Z_m},Z_m=\sum_{i=1}^Nw_{m,i}
$$
Set initial values to be uniformly distributed, $w_{1,i}=\frac{1}{N}​$. When progressing to the next model, $\phi_{m+1}​$, we treat the new weights as following:

- For the rightly classified samples, multiply its weight by some constant $d_m​$.
- For the wrongly classified samples, divide its weight by the same constant $d_m$.

Suppose the error/rate of correctness of the next model is randomly distributed, there is:
$$
\sum_{i=1}^Nw_{m+1,i}I(y_i\ne\phi_m(x_i))=\sum_{i=1}^Nw_{m+1,i}I(u_i=\phi_m(x_i))\\\Rightarrow\sum_{i=1}^Nw_{m,i}d_mI(y_i\ne\phi_m(x_i))=\sum_{i=1}^Nw_{m,i}/d_mI(u_i=\phi_m(x_i))\\\Rightarrow d_m=\sqrt{\frac{1-\epsilon_m}{\epsilon_m}}>1
$$
The new weights are computed in this form:
$$
w_{m+1,i}=\frac{w_{m,i}\exp(-\alpha_my_i\phi_m(x_i))}{Z_{m+1}},Z_{m+1}=\sum_{i=1}^Nw_{m+1,i},\alpha_m=\frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}
$$

The final classifier is:
$$
f(x)=sgn(\sum_{m=1}^M\alpha_m\phi_m(x))
$$

### Proof

We hereby prove that with more models, all kinds of errors may be reduced.
$$
\frac{dERR_{train}(f)}{dM}\le0
$$
Express $w_{M+1,i}$:
$$
w_{M+1,i}=w_{M,i}\frac{\exp(-\alpha_My_i\phi_M(x_i))}{Z_{M+1}}=w_{1,i}\frac{\exp(-y_i\sum_{m=1}^M\alpha_m\phi_m(x_i))}{\prod_{m=1}^MZ_{m+1}}=w_{1,i}\frac{\exp(-y_if(x_i))}{\prod_{m=1}^MZ_{m+1}}
$$
Express $Z$, with $w_{1,i}=\frac{1}{N}$:
$$
\prod_{m=1}^MZ_{m+1}=\frac{1}{N}\sum_{i=1}^{N}\exp(-y_if(x_i))
$$
Express training error:
$$
ERR_{train}(f(x))=\frac{1}{N}\sum_{i=1}^NI(y_i\ne sgn(f(x_i)))\le\frac{1}{N}\sum_{i=1}^N\exp(-y_if(x_i))=\prod_{m=1}^MZ_{m+1}
$$
Training error is bounded by exponential loss of the model.

![1554632021011](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1554632021011.png)

Express more of $Z$:
$$
Z_{m+1}=\sum_{i=1}^Nw_{m,i}d_mI(y_i\ne\phi_m(x_i))+\sum_{i=1}^Nw_{m,i}/d_mI(y_i=\phi_m(x_i))=2Z_m\sqrt{\epsilon_m(1-\epsilon_m)}
$$
Plug in training error:
$$
\frac{1}{N}\sum_{i=1}^N\exp(-y_if(x_i))=\prod_{m=1}^MZ_{m+1}=2^M\prod_{m=1}^M\sqrt{\epsilon_m(1-\epsilon_m)}
$$
Tell whether it increase or decrease with respect to $M​$:
$$
\frac{2^{M+1}\prod_{m=1}^{M+1}\sqrt{\epsilon_m(1-\epsilon_m)}}{2^M\prod_{m=1}^M\sqrt{\epsilon_m(1-\epsilon_m)}}=2\sqrt{\epsilon_{M+1}(1-\epsilon_{M+1})}\le1
$$
Thus, the problem of the contradiction between over fitting and under fitting is perfectly solved.

## Gradient Boosting

### General Process

Apply the idea of gradient descend here, we substitute the gradient with a new model.
$$
f_m(x)=f_{m-1}(x)+\alpha_m\phi_m(x)
$$

- Initialize $f_0(x)=\min_f\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_1))$.
- for $m=1:M$,
  - Compute gradient residual using $r_{m,i}=\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)},f=f_{m-1}$.
  - Use the weak learner which minimizes $\sum_{i=1}^N(r_{m,i}-\phi_m(x_i))^2$.
  - Update $f_m(x)=f_{m-1}(x)+\eta\phi_m(x)$.
- return $f(x)=f_M(x)$.

### AdaBoost,

- Plug in exponential loss, $L=\exp(-yf(x))$.
- Plug in target function, $J(f)=\sum_{i=1}^NL(f(x_i),y_i)$.
- Negative gradient of step m,

$$
-\frac{\partial J(f_m)}{\partial f_m}=\sum_{i=1}^Ny_i\exp(-y_if_m(x_i))
$$
The mth model must reduce the target function to the most. The process is terminated when all models are counted within the ensemble model.

### Proof

We hereby try to prove this conclusion.
$$
\frac{\partial J(f)}{\partial \alpha_m}=0\Rightarrow\alpha_m=\frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}
$$

Express $J$:
$$
J(f)=\sum_{i=1}^N\exp(-y_i(f_{m-1}(x_i)+\alpha_m\phi_m(x_i)))=\exp(-\alpha_m)\sum_{y_i=\phi_m(x_i)}w_{m,i}+\exp(\alpha_m)\sum_{y_i\ne\phi_m(x_i)}w_{m,i}
$$
Take derivative and the result is obvious.

### L2 Loss

$$
J(f)=\sum_{i=1}^N(f(x_i)-y_i)^2,\frac{\partial J(f)}{\partial f}=2\sum_{i=1}^N(f(x_i)-y_i)
$$

- Initialize, $f_0(x)=\bar y$.
- mth step, $r_{m,i}=-2\sum_{i=1}^N(f(x_i)-y_i)$

### Shrinkage

For a larger amount of models, add an extra coefficient $\eta$, also known as learning rate.
$$
f_m(x)=f_{m-1}(x)+\eta\alpha_m\phi_m(x)
$$
