# Basic Network

## Perceptron

![img](https://upload-images.jianshu.io/upload_images/11345863-4feca33b55dd2350.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/397/format/webp)

Perceptron is the smallest unit of a network. As is shown in the picture, a perceptron consists of the input part and the output part. The input is implemented by weighted sum.
$$
logits=Wx+b=b+\sum_{i=1}^nw_nx_n=\sum_{i=0}^nw_nx_n\quad(w_0=b)
$$
The output is implemented by step function call.
$$
output=f(logits)=f(b+\sum_{i=1}^nw_nx_n)\\
f(x)=\begin{cases}
1 & x>0\\
0 & x\le 0
\end{cases}
$$
For a single perceptron, it can be fitted into some boolean function, such as And and Or. There may be a proof for such possibility.

Such implementation of a boolean function can be taken that the function is told by whether a linear inequality holds.

| x_1  | x_2  | output |
| :--: | :--: | :----: |
|  0   |  0   |   0    |
|  0   |  1   |   0    |
|  1   |  0   |   0    |
|  1   |  1   |   1    |

The table shows the property of And. If there is a linear inequality which satisfy the table, such that:
$$
a_1x_1+a_2x_2+b>0
$$
Plug in the numbers:
$$
\begin{cases}
b<0\\
a_1+b<0\\
a_2+b<0\\
a_1+a_2+b>0
\end{cases}
$$
It is obvious that the last one can be derived by the others and there is no contradiction among the inequalities. Same thing happens for Or:
$$
\begin{cases}
b<0\\
a_1+b>0\\
a_2+b>0\\
a_1+a_2+b>0
\end{cases}
$$

However, for Xor function, a single linear inequality cannot fully describe it.
$$
\begin{cases}
b>0\\
a_1+b<0\\
a_2+b<0\\
a_1+a_2+b>0
\end{cases}
$$
If substitute the second and third inequality into the forth, it derives a contradiction.

## Full-connected Network

Many perceptrons, or neutrons altogether form a complete network. A layer of a network consists of a series of perceptrons, whose input comes from all of the outputs of the previous layer and output goes to every perceptron on the next layer.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png)

Each perceptron on the input layer only has a single input, while each perceptron on the output layer only has a single output.

## Propagation

### Forward Propagation

For the ith layer of a network:
$$
logits^{(i)}(h^{(i-1)})=b^{(i)}+\begin{pmatrix}
W^{(i)}_{1}h^{(i-1)}\\
W^{(i)}_{2}h^{(i-1)}\\
\vdots\\
W^{(i)}_{j}h^{(i-1)}\\
\end{pmatrix}=b^{(i)}+W^{(i)}h^{(i-1)},h^{(i)}=\sigma(logits^{(i)})
$$
where $logits$ signifies the computational result of the layer before the activation function, $W_i$ signifies the coefficient matrix of the layer, not $W$, $h_{i}$ signifies the final output of the layer. Note that $W_i$ signifies a matrix, not a vector. For input layer, the input must be signified.
$$
h^{(i)}=FP^{(i-1)}(h^{(i-1)})=FP^{(1)}(FP^{(2)}(...FP^{(i)}(h^{(0)}))),\hat{y}=h^{(last)}
$$

### Backward Propagation

It is not enough to only have a network which gives an output. The model has to be trained somehow to fit itself according to the real world. The goal, like other machine learning model, is to minimize the difference between the real world and the output. There can be L2 cost or L1 cost. Take L2 cost as an example, with total number of hidden layers to be $L$, $i$ for the index of perceptron, and $j$ for the index of the coefficient of the perceptron:
$$
Cost(x)=\frac{1}{2}(y-\hat{y})^2,logits^{(l)}_{i}=W^{(l)}_{i1}h^{(l-1)}_{1}+W^{(l)}_{i2}h^{(l-1)}_{2}+...
$$

### Algorithm

For the last layer:
$$
\begin{align*}&\frac{\partial Cost}{\partial W^{(L)}_{jk}}=\frac{\partial Cost}{\partial logit^{(L)}_j}\times \frac{\partial(W^{(L)}_{j1}h^{(L-1)}_{1}+W^{(L)}_{j2}h^{(L-1)}_{2}+...)}{\partial W^{(L)}_{jk}}=\delta^{(L)}_{j}h^{(L-1)}_j
\end{align*}
$$
For other layers:
$$
\delta^{(l)}_i=\frac{\partial Cost}{\partial logit^{(l)}_i}=\sum_n(\frac{\partial Cost}{\partial logit^{(l+1)}_n}\times \frac{\partial logit^{(l+1)}_n}{\partial logit^{(l)}_i})=\sum_n \delta_n^{(l+1)}W^{(l+1)}_{ni}\sigma'(logit_i^{(l)}),\frac{\partial Cost}{\partial b^{(l)}_i}=\delta_i^{(l)}
$$
In general:
$$
\begin{align*}
\delta^{(L)}&=(\nabla_yCost)^T\sigma'(logit^{(L)})\\
\delta^{(l)}&=((W^{(l+1)})^T\delta^{(l+1)})^T\sigma'(logit^{(l)})\\
\frac{\partial Cost}{\partial b^{(l)}_i}&=\delta_i^{(l)}\\
\frac{\partial Cost}{\partial W^{(l)}}&=\sum_n \delta_n^{(l+1)}W^{(l+1)}_{ni}\sigma'(logit_i^{(l)})\\
\end{align*}
(l<L)
$$

### Problems

##### Gradient Disappearance

In the process of propagation, forward and backward, the perceptrons far from the output layer has less effect on the output layer. When the number of layers grows too large, due to the accuracy of float number representation in computer systems, the previous layers of the network barely changes according to the process of training.

##### Zig-zag

When using the gradient descend method, one step of descending might be too large or too small, causing the next step to somehow go back a little to the original pass, and so does the next steps behave.

![The gradient descent algorithm in action. (1: contour)](https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Gradient_ascent_%28contour%29.png/350px-Gradient_ascent_%28contour%29.png)

## Activation Function

The simplest activation function is the step function.
$$
\sigma(x)=\begin{cases}
1 & x>0\\
0 & otherwise
\end{cases}
$$
Such function is neither differentiable or continuous, making us finding other supplement. In general, $\sigma(x)$ has to satisfy the following terms:

1. Has at least an upper or lower bound.
2. Continuous.
3. Undifferentiable at countable places.

The most popular choice is the sigmoid function:
$$
sigmoid(x)=\frac{1}{1+e^{-x}},sigmoid'(x)=sigmoid(x)(1-sigmoid(x))
$$
However, there is a problem for the derivative:
$$
sigmoid'(x)\le \frac{1}{4}
$$
With each multiplication, the result becomes closer to 0, and, in the end, under the computer system’s floating point accuracy, becomes 0. Moreover, the function value is always positive, which initially causes the zig-zag problem. Through a simple transformation, the problem could be easily solved.
$$
\tanh(x)=2sigmoid(2x)-1=\frac{e^x-e^{-x}}{e^x+e^{-x}},\tanh'(x)=4sigmoid'(2x)\le 1
$$
Still, like sigmoid, there is a saturated region in $\tanh​$. Thereby still exists the problem of gradient disappearance. To eliminate gradient disappearance, there must be a region in the function where it is always not saturated. RELU is a good example.
$$
ReLU(x)=\begin{cases}
x & x>0\\
0 & otherwise
\end{cases}
$$
The function ignores the zig-zag problem, for it alway gives a number of either 0 or larger than 0. Further alteration:
$$
\begin{align*}
PReLU(x)&=\begin{cases}
x & x>0\\
\alpha x & otherwise
\end{cases}\\
MaxOut(x)&=max\left\{w_1^Tx+b_1,w_2^Tx+b_2,...,w_n^Tx+b_n,\right\}\\
ELU(x)&=\begin{cases}
x & x>0\\
\alpha (e^x-1) & otherwise
\end{cases}\\
SELU(x)&=selu_{scale}\begin{cases}
x & x>0\\
selu_\alpha (e^x-1) & otherwise
\end{cases},\\
&selu_\alpha=1.6732632423543772848170429916717\\
&selu_{scale}=1.0507009873554804934193349852946\\
Swish(x)&=x\times sigmoid(x)
\end{align*}
$$

### Batch Normalization (BN)

In order to solve the problem of saturated region, we add some preparation on the input. After the input layer, we insert a new layer of Batch Normalization layer to achieve such goal.

Separate the input into multiple mini-batches, $B=\left\{x_1,...,x_m\right\}$. For each of these batches, apply the following operation and give the output $\left\{y_i=BN_{\gamma,\beta}(x_i)\right\}$. $\gamma,\beta$ are trained along with other variables. They must be taken into consideration, for a balance between the saturated region and the unsaturated region must be somehow maintained. $\epsilon​$ is a relatively small number to prevent negative number under a square root, due to the accuracy of floating points under computer systems.
$$
\begin{align*}
\mu_B&=\frac{1}{m}\sum_{i=1}^mx_i\\
\sigma_B^2&=\frac{1}{m}\sum_{i=1}^m(x_i-\mu_B)^2\\
\hat{x}_i&=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}\\
y_i&=\gamma x_i+\beta\\
\end{align*}
$$
Forward Propagation:
$$
\begin{align*}
x\text{ as input}&\\
s_1&=W_{h_1}x\text{,   no bias term}\\
s_2&=\frac{s_1-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}\\
s_3&=\gamma s_2+\beta\\
h_1&=ReLU(s_3)\\
h_1\text{ as final output}&
\end{align*}
$$
Backward Propagation:
$$
\begin{cases}
\frac{\partial Cost}{\partial \hat{x}_i}&=\frac{\partial Cost}{\partial y_i}\gamma\\
\frac{\partial Cost}{\partial \sigma_B^2}&=-\frac{1}{2}(\sigma_B^2+\epsilon)^{-\frac{3}{2}}\sum_{i=1}^m(x_i-\mu_B)\\
\frac{\partial Cost}{\partial \mu_B}&=(\sum_{i=1}^m\hat{x}_i\frac{-1}{\sqrt{\sigma_B^2+\epsilon}})-2\frac{\partial Cost}{\partial \sigma_B^2}\frac{\sum_{i=1}^m(x_i-\mu_B)}{m} \\
\frac{\partial Cost}{\partial x_i}&=\frac{\partial Cost}{\partial \hat{x}_i}\frac{1}{\sqrt{\sigma_B^2+\epsilon}}+\frac{\partial Cost}{\partial \sigma_B^2}\frac{2(x_i-\mu_B)}{m}+\frac{1}{m}\frac{\partial Cost}{\partial \mu_B} \\
\frac{\partial Cost}{\partial \gamma}&=\sum_{i=1}^m\frac{\partial Cost}{\partial y_i}\hat{x}_i \\
\frac{\partial Cost}{\partial \beta}&=\sum_{i=1}^m\frac{\partial Cost}{\partial y_i}
\end{cases}
$$

## Output Layer

Under different goals of network, we might demand different type of output.

![1551016878189](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1551016878189.png)

The softmax layer consists of a series of neurons with only a single input and output. Each neuron computes the possibility of the model’s input belonging to a certain category.
$$
softmax(x_i)=\frac{e^{x_i}}{\sum_{i=1}^Ne^{x_i}}
$$
The cost function of classifying model is still the cross entropy loss.
$$
H(X)=-\sum_{i=1}T^Ny(x_i)\ln p(x_i)
$$

## Loss Function

### Classifying Loss

0-1 loss:
$$
L(Y,f(Y))=\begin{cases}
1 & Y\ne f(X)\\
0 & Y=f(X)\\
\end{cases}
$$
Exponential Loss:
$$
L(Y,f(X))=e^{-Yf(X)}
$$
Hinge Loss:
$$
L(Y,f(X))=\max{0,1,-Yf(X)}
$$
Cross Entropy Loss:
$$
L(Y,f(X))=-Y\ln(f(X))
$$


### Regression Loss



## Over Regression

### Dropout

In order to improve the behavior of a deep network, when training a network, for each layer, we drop out every neuron with a constant probability $p​$. Therefore, the number of the remaining neurons is Bernoulli distributed. 

![1551015762087](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1551015762087.png)

The output of the layer should be:
$$
\widetilde{h}^{(l)}=r^{(l)}h^{(l)},r^{(l)}\sim Bernoulli(p)
$$
