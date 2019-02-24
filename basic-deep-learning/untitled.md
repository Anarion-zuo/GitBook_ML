# Full-connected Network

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

## Network

Many perceptrons, or neutrons altogether form a complete network. A layer of a network consists of a series of perceptrons, whose input comes from all of the outputs of the previous layer and output goes to every perceptron on the next layer.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png)

Each perceptron on the input layer only has a single input, while each perceptron on the output layer only has a single output.

## Forward Propagation

For the ith layer of a network:
$$
logits_i(h_{i-1})=b_{i}+\begin{pmatrix}
W_{i1}h_{i-1}\\
W_{i2}h_{i-1}\\
\vdots\\
W_{ij}h_{i-1}\\
\end{pmatrix}=b_i+W_ih_{i-1},h_i=f(logits_i)
$$
where $logits$ signifies the computational result of the layer before the trigger function, $W_i$ signifies the coefficient matrix of the layer, not $W$, $h_{i}$ signifies the final output of the layer. Note that $W_i$ signifies a matrix, not a vector. For input layer, the input must be signified.
$$
h_i=FP_{i-1}(h_{i-1})=FP_1(FP_2(...FP_i(h_0)))
$$


