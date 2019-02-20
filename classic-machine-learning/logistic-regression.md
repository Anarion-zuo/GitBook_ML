# Logistic Regression

## Loss Function

### Distribution

If the model is designed to give output of telling whether the input falls into some kind of a category, the distribution of the output should be Bernoulli.

$$
y\sim B(1,p),p(y;\mu)=\mu^y(1-\mu)^{(1-y)},Ey=\mu
$$

| $$y$$ | 0 | 1 |
| :---: | :---: | :---: |
| $$p$$ | $$1-\mu$$ | $$\mu$$ |

The means of the output is some value between 0 and 1, according to the possible value of y. Therefore, there must be some transformation to bound the means to that interval. One of the most commonly used function is the Sigmoid function.

$$
\sigma(x)=\frac{1}{1+e^{-x}},\sigma'(x)=\sigma(x)(1-\sigma(x)),\mu(x)=\sigma(f(x))
$$

The odds of the distribution:

$$
f(x)=\ln\frac{p(y=1|x)}{p(y=0|x)}
$$

For linear models:

$$
\mu(x)=\sigma(W^Tx)
$$

The rule of deciding the output is simple, which is to check which probability is larger.

1. $$f(x)>0,y=1$$
2. $$f(x)<0,y=0$$ 
3. $$f(x)=0$$ , special treatment.

![Simple 2-var Case](../.gitbook/assets/image%20%282%29.png)

### 0/1 Loss and Surrogate

Given that the output is binary, the loss function can be binary.

$$
L(y,\hat{y})=\begin{cases}0 & y=\hat{y}\\
1 & y\ne\hat{y}
\end{cases}
$$

A surrogate method has to be found, for we cannot deal with undifferentiable functions. The most intuitive idea is the maximum likelihood estimation.

$$
l(\mu)=\sum_{i=1}^N(y_i\ln \mu(x_i)+(1-y_i)\ln(1-\mu(x_i)))
$$

Loss function:

$$
L(y,\mu(x))=-l(\mu)
$$

Simillarly, the regular term can be L1 or L2. There must be a regular term in Logistic Regression model, for the complexity of the model is too much. Therefore, in sklearn, the target function is:

$$
J(W;C)=C\sum_{i=1}^NL(y_i,\mu(x_i;W))+R(W)
$$

## Better Solution for Logistic Model

### Derivatives

$$
g(w) = \nabla J(w) = \sum_{i=1}^N(\mu_i - y_i)x_i = X^T(\mu - y)\\
H(w) = \frac{\partial g(w)}{\partial w} = \sum_{i=1}^N x_i^T\mu_i(1-\mu_i)x_i = X^TSX,S = diag(\mu_i(1-\mu_i))
$$

### Newton's method

$$
w_{t+1} = w_t - H^{-1}(w_t)g(w) = (X^TS_tX)^{-1}X^TS_tz_t
$$

The formula is similar to Least Square method. Hence, Newton's method in Logistic model is also called Iteratively Reweighted Least Square \(IRLS\).

## Multi-classifying

### One-vs-Rest\(OvR\)

Namely, take one of the categories as one category and the rest as another.

### Multinoulli/Categorical Distribution

Bernoulli distribution has only 2 kinds of output whereas Multinoulli has more kinds of output.

| $X$ | $x\_1$ | $x\_2$ | $...$ | $x\_N$ |
| :---: | :---: | :---: | :---: | :---: |
| $\theta$ | $\theta\_2$ | $\theta\_3$ | $...$ | $\theta\_N$ |

Write the distribution continuously.

$$
\sum_{c=1}^C\theta_c=1,Cat(y,\theta)=\prod_{c=1}^C\theta^{y_c}_c
$$

Also known as one-hot encoding.

### Softmax

Expend the idea of sigmoid function to be softmax function.

$$
\sigma(z_c) = \frac{e^{z_c}}{\sum_{c'=1}^C e^{z_{c'}}},\mu_c=p(y=c|x,w)=\frac{e^{w_c^Tx}}{\sum_{c'=1}^C e^{w_{c'}^Tx}}
$$

Maximum likelihood estimation:

$$
l(M) = \sum_{i=1}^N\ln\prod_{c=1}^C\mu_{ic}^{y_{ic}}=\sum_{i=1}^N\sum_{c=1}^Cy_{ic}\ln\mu_{ic}
$$

Softmax Loss:

$$
J(w) = -l(M) = \sum_{i=1}^N(\sum_{c=1}^Cy_{ic}w_c^Tx-\ln\sum_{c=1}^Ce^{w_{c'}^Tx})
$$

## Unbalanced Data Sample

Often, some categories give less data sample than other categories. When the difference of the amount of data sample is too significant, we may do something to balance the categories.

### Undersampling

Randomly take a part of the larger sample as a sample, which might cause some damage of information and some error.

### OverSampling

Randomly repeatedly rake a part of the smaller sample as a sample, which might cause the weight of certain parts of the sample to be unjustly magnified and the model to lean upon those parts.

### EasyEnsemble

Generate n samples by n times of under-sampling from larger samples, by which generate n models. The output of the final model is the average of these models.

### BalanceCascade

Generate a sample as large as the small sample by under-sampling from larger samples. Generate a model with this sample and other smaller samples. Select those ones of mistakes and fix the model with other part of the larger sample.

### Synthetic Minority Over-sampling Technique \(SMOTE\)

* Select a data frame in a smaller sample with index i and vector $x\_i$.
* Find K neighbours of $x_i$ in the smaller sample $x_{i\(near\)},near\in{1,...,K}$.
* Randomly select a sample in the neighbours $x\_{i\(nn\)}$. And generate a random number between 0 and 1, $\zeta$,

  $$
  x_{i1} = (1-\zeta)x_i + \zeta x_{i(near)}
  $$

  whereas $x_i$ is a point on the straight line between all of the $x_{i\(nn\)}$ as an insertion.

* SMOTE algorithm can prevent errors from over fitting. However it is not so valid for high dimensional cases and might cause over lap among categories. In order to eliminate the disadvantage of SMOTE, we have Borderline-SMOTE and many other fixing method.

  **Cost-Sensitive Learning**

  Build a cost matrix:

  |  | Prediction wrong | Prediction right |
  | :---: | :---: | :---: |
  | Predict 0 | $C\_{00}$ | $C\_{01}$ |

  Predict 1 \| $C_{10}$ \| $C_{11}$ \|

  **Class Weight**

* Regardless of class weights;
* Balanced mode: Compute class weight according to the amount of certain data samples;
* Manually set.

## Evaluation

### Loss Function

The value of the loss can be taken as an approach of evaluation. The less the better.

$$
\ln J(Y,P) = -\frac{1}{N}\sum_{i=0}^N\sum_{c=0}^Cy_{ic}\ln p_{ic}
$$

* $C$ for the number of categories. $y_i$ for the vector of one-hot encoding. $p_{ic}$ for the probability for the output to be the $i$th sample of category $c$.

  **Zero-one Loss**

  The zero-one loss function's average output over all test result can serve as error rate.

  $$
  \frac{1}{N}\sum_{i=0}^NI(\hat{y}_i \ne y_i)
  $$

  Same goes for accuracy score:

  $$
  \frac{1}{N}\sum_{i=0}^NI(\hat{y}_i = y_i)
  $$

  **Hinge Loss**

  $$
  \frac{1}{N}\sum_{i=0}^NI(1 - \hat{y}_iy_i)
  $$

  **Confusion Matrix**

  $a\_{ij}$ stands for having an object belongs to $i$th category classified to $j$th property by the model. A good enough model should have a confusion matrix with many large numbers on the diag.

  **ROC/AUC\(OvR\)**

  |  | $\hat{y} = 1$ | $\hat{y} = 0$ | $\Sigma$ |
  | :---: | :---: | :---: | :---: |
  | $y = 1$ | True Positive\(TP\) | False Negative\(FN\) | $N\_+$ |
  | $y = 0$ | False Positive\(FP\) | True Negative\(TN\) | $N\_-$ |
  | $\Sigma$ | $\hat{N}\_+$ | $\hat{N}\_+$ |  |

  $$
  Precision = \frac{TP}{\hat{N}_+},
  TPR = Recall = \frac{TP}{N_+},
  FPR = \frac{FP}{N_-}\\
  F1 = 2\frac{Precision \times Recall}{Precision + Recall} = \frac{TP}{2TP + FP + FN}
  $$

  Often, when we have high precision, we have low recall. Backward also. Hence, we need a balanced factor of these 2 $F1$.

  $$
  Matthews\_Correlation\_Coefficient = MCC = \\ \frac{TP\times TN - FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}\\
  MCC = \begin{cases}
    1 & \Rightarrow perfect\quad model\\
    0 & \Rightarrow random\quad model\\
    -1 & \Rightarrow kidding\quad model(\text{outputs are exactly opposite to the real world})
  \end{cases}
  MCC\in [-1,1]
  $$

  When the samples are extremely unbalanced, the accuracy score is not a valid evaluation. We should apply the MCC and other methods.

In the process of classifying, for each returned probability value from the model, we deem the object to be in a certain category according to a certain threshold. Namely, if the probability is larger than the threshold, we deem it to be in one category, and if not, the other. In order to find the best threshold value, we plug and try different values into the model and draw ROC curve. The closer the curve is to the left up angle, the better the model is. However, the shape of the curve can only be told by human eyes, not for machines to compare its magnitude. In order to have something for the machine to deal with, we compute the area under the curve to determine whether the model is good enough accurately. Another advantage of ROC is that the curve does not change easily according to the samples. 

![ROC](../.gitbook/assets/image%20%284%29.png)

### Precision and Recall \(PR\)\(OvR\)

The PR curve is applied when estimating rare events such as recommending system and information searching. Specifically, it is when $N\_-$ is very small and ROC is too close to x-axis. The closer the curve is to the right up angle, the better the model is. 

![PR](../.gitbook/assets/image%20%283%29.png)

 Similarly, a number can also be extracted from the curve.

$$
AP = \sum_{k=0}^nP(k)\Delta R(k),mAP=\frac{1}{K}\sum_{i=1}^KAP_i
$$



