# Supporting Vector Machine

## Idea

### Basis

For any dimension of sample, there may be a line or surface dividing the sample space into 2 parts, which is described in a linear equation.
$$
w^Tx+b=0
$$
To decide whether a point of sample is on the right or left hand side of the surface, we introduce a linear function, holding the output value of the equation’s left hand side.
$$
f(x)=w^Tx+b
$$
The specific class of a sample is, thereby, told by the sign of $f(x)$. Further more, a surface can be described in the form related to distance.
$$
f(x)=w^T(x_p+\gamma\frac{w}{||w||_2})+b=\gamma||w||_2
$$
where $\gamma$ is the “signed” distance from a specified sample $x$ to the surface. Specially, for the origin:
$$
\gamma_0=\frac{b}{||w||_2}
$$
For classification models, the outputs $y$ generally consist of 1s and -1s. Therefore, the “real” distances, instead of “signed” ones, are:
$$
\frac{yf(x)}{||w||_2}
$$
A good enough classifying machine, in some extend, has a large margin between the classes, namely, the distance of the samples on both sides of the bounding surface nearest to the surface to the surface must be as large as possible. Such is also stated as the margin should be as large as possible. In the SVM case, for convenience, we set the margin to be the region between $f(x)=1$ and $f(x)=-1$. According to the previous deduction result, the distance between the boundary of the margin is given by:
$$
\frac{2}{||w||_2}
$$

### Loss Function

Obviously, the magnitude of weight must be as little as possible. Hence, the goal of SVM becomes:
$$
\begin{align*}
&\min_{w,b}\quad\frac{1}{2}||w||_2^2\\
&s.t.\quad y_i(w^T+b)\ge1,i=1,...,N
\end{align*}
$$
The idea above works for the samples totally linearly separable. When dealing with real problems, we may tolerate some error. Therefore, the margin may be as large as possible under some tiny error. The goal becomes:
$$
\begin{align*}
&\min_{w,b}\quad\frac{1}{2}||w||_2^2+C\sum_{i=1}^N\xi_i\\
&s.t.\quad y_i(w^Tx+w_0)\ge1-\xi_i,i=1,...,N,\xi_i\ge0
\end{align*}
$$
Under the special case of SVM, we define that when $y_i(w_0+w^Tx_i)\ge1$, there is $\xi_i=0$. And if not so, $\xi_i=1-y_i(w_0+w^Tx_i)$. Therefore, the loss function is set to be:
$$
L_{Hinge}(y,\hat{y})=\begin{cases}
0 & y\hat{y}\ge1\\
1-y\hat{y} & otherwise
\end{cases}
$$
And the target function is set to be:
$$
J(w;C)=\frac{1}{2}||w||^2_2+C\sum_{i=1}^NL_{Hinge}(y_i,f(x_i;w))
$$
Or, in special cases:
$$
\begin{align*}
&J(w;C)=\frac{1}{2}||w||^2_2+C\sum_{i=1}^N\xi_i\\
&s.t.\quad1-\xi_i-y_i(w^Tx_i+w_0)\le0,i=1,...,N\\
&\qquad-\xi_i\le0,i=1,...,N
\end{align*}
$$


## Solution

### Lagrange Method

Rearrange the target function to the form of Lagrange.
$$
L=\frac{1}{2}||w||_2^2+C\sum_{i=0}^N\xi_i-\sum_{i=1}^N\alpha(-1+\xi_i+y_i(w^Tx_i+w_0))-\sum_{i=1}^N\mu_i\xi_i
$$
With terms of:
$$
\alpha_i,\mu_i,\xi_i\ge0
$$

### Duality

For a typical minimization problem:
$$
\min_x\theta_P(x)=\min_x\max_{\alpha,\beta:\alpha\ge0}L(x,\alpha,\beta)
$$
where subscription $P$ signifies the original problem. Let $p^*$ be the solution. The “dual” version of this problem is defined as:
$$
\max_{\alpha,\beta:\alpha\ge0}\theta_D(\alpha,\beta)=\max_{\alpha,\beta:\alpha\ge0}\min_xL(x,\alpha,\beta)
$$
where subscription $D$ signifies the dual version. Let $d^*$ be the solution. It may be easy to prove:
$$
d^*\le p^*
$$
An intuitive way of understanding is to take these pair of solution as the maximum in the minimums and minimums in the maximums. The precondition for the equality to hold is:

1. The main function $f$ and inequality boundaries $g_i$ are convex. The equality boundaries $h_i$ are linear functions, $h_j(x)=a^Tx+b$.
2. $g_i$ are “valid”, namely, contain some region, or $\exist x,g_i(x)\le0$.

There has to exist $x^*$ as the solution to both problems, $\alpha^*,\beta^*$ as the solution to dual problem, and $d^*=p^*$.

Also, the $x^*,\alpha^*,\beta^*$ satisfy the Karush-Kuhn-Tucker (KKT) conditions.

### KKT

$$
\frac{\partial L(x^*,\alpha^*,\beta^*)}{\partial x}=0
$$

$$
\frac{\partial L(x^*,\alpha^*,\beta^*)}{\partial \beta_j}=0,1\le j\le L
$$

$$
\alpha_i^*\ge0,1\le i\le K
$$

$$
g_i(x^*)\le0,1\le i\le K
$$

Specially, the dual complementary condition:
$$
a_i^*g_i(x^*)=0,1\le i\le K
$$

Plug in for Lagrange problem:
$$
\begin{align*}
&\frac{\partial L}{\partial w}=0\Rightarrow w=\sum_{i=1}^N\alpha_iy_ix_i\\
&\frac{\partial L}{\partial b}=0\Rightarrow\sum_{i=1}^N\alpha_iy_i=0\\
&\frac{\partial L}{\partial\xi_i}=0\Rightarrow C=\alpha_i+\mu_i
\end{align*}
$$
Therefore, the target function is:
$$
L(\alpha)=\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^Tx_j,0\le\alpha_i\le C,\sum_{i=1}^N\alpha_iy_i=0
$$
After $\alpha$ is obtained, the other factors can be obtained by it.
$$
w=\sum_{i=1}^N\alpha_iy_ix_i,b=\frac{1}{N_S}\sum_{m\in S}(y_m-\sum_{m'\in S}\alpha_{m'}y_{m'}x_m^Tx_{m'})
$$
The model is:
$$
f(x)=\sum_{i=1}^N\alpha_iy_ix_i^Tx+b,\hat{y}=sgn(f(x))
$$

### Dual Complementary Condition

$$
a_i^*g_i(x^*)=0,1\le i\le K
$$

It can be found from the formula that when $a_i^*$ is 0, the sample is weightless, while when $g_i^*​$ is 0, the sample is quite weighted. The weighted samples are called the supported vectors.

## Kernel

In order to be better than the behavior of a simple linear model, we introduce kernel functions into our model, so that some unlinearity may be introduced by it. The kernel function is represneted as:
$$
\phi(x)=(x_1^2,\sqrt{2}x_1x_2,x_2^2)
$$
or other form of transformation. For all of the expressions mentioned above, $x$ should be changed into $\phi(x)$. For polynomial cases, when the number of dimensions is to large, the time complexity of constructing kernel or dot product may be too large. Therefore, we compute the dot product beforehand.
$$
k(x,x')=\phi(x)^T\phi(x')
$$

### Polynomial

$$
k(x,x')=(\gamma x^Tx'+r)^M
$$

### RBF

$$
k(x,x')=\exp(-\frac{1}{2}(x-x')^T\Sigma^{-1}(x-x'))
$$



