# Logistic Regression

## Loss Function

### Distribution

If the model is designed to give output of telling whether the input falls into some kind of a category, the distribution of the output should be Bernoulli.

$$
y\sim B(1,p),p(y;\mu)=\mu^y(1-\mu)^{(1-y)},Ey=\mu
$$

| $$y$$  | 0 | 1 |
| :---: | :---: | :---: |
| $$p$$  | $$1-\mu$$ | $$\mu$$  |

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

