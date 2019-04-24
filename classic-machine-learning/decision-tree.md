# Decision Tree

A decision tree is a tree, not necessarily a binary tree. At every non-leaf node, as we go along the tree, we decide which children do we go next, according to the input information and terms at the node. There are many ways of constructing a decision tree.

## ID3

For a C-class classification problem,
$$
p(Y=c)=\frac{1}{D}\sum_{i\in D}I(y_i=c)
$$

The entropy of the samples is:
$$
H(D)=-\sum_{c=1}^Cp(Y=c)\ln p(Y=c)
$$
If we split the set of the sample into $V$ subsets, the entropy becomes:
$$
H_X(D)=\sum_{v=1}^V\frac{|D_v|}{|D|}H(D_v)
$$
There is gain in entropy as we split, for new tags for the new subsets are new information. The gain, provided in splitting feature $X$, given to tag/class $Y$ is:
$$
gain_X(D)=H(D)-H_X(D)
$$
Compute $H_L$ for each of the features, find the one holding maximum $gain$ and work on it. Split the chosen feature according to the potential values the feature may hold, and work on the other features. If the feature has many potential features, the children of the node may get too many and yet contribute little to the model.

## C4.5

Instead of focusing on the features with more children, C4.5 tends to focus on the features with more information gain.
$$
split\_info_X(D)=-\sum_{v=1}^V\frac{|D_v|}{|D|}\log_2\frac{|D_v|}{|D|},gain\_ratio_X(D)=\frac{gain_X(D)}{split\_info_X(D)}
$$
We must keep it in mind that the gain of information should be at the cost of having too many children. Still C4.5 constructs children in the same way as ID3, taking all potential values into consideration.

## CART

A CART is a binary tree, constructed by a recursing process. Instead of constructing children with all choices of potential values, we set boundaries to between the values.

![1555732113817](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1555732113817.png)

The result of the model often looks like step functions, often too easy to be over fitting. The constructing process has to be strictly under control to prevent over fitting, such as stopping in advance, pruning, and other ensemble learning procedures.

![1555732214931](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1555732214931.png)

### Construction

- For a training set $D$, splitting feature $j$, splitting threshold $t_m$, evaluating splitting operation $\theta(j,t_m)$, split the set into a left-hand side and right-hand side $D_L,D_R$.
  - $D_L(\theta)=\{(x_i,y_i)|x_{ij}\le t_m\}​$
  - $D_R(\theta)=\{(x_i,y_i)|x_{ij}> t_m\}$
- Split to make the children as “pure” as possible.

$$
G(D,\theta)=\frac{N_{left}}{N_m}H(D_L(\theta))+\frac{N_{right}}{N_m}H(D_R(\theta)),\theta^*=\arg\min_\theta G(D,\theta)
$$

The degree of “impure” is measured as above. The function describing the degree of impure $H​$ is related to the training set.

### Discrete Input / Classification

Many choices of $H$ we may have here. Suppose the rate of each class is represented as:
$$
\hat\pi_c=\frac{1}{D}\sum_{i\in D}I(y_i=c)
$$


Rate of error:
$$
H(D)=\frac{1}{D}\sum_{i\in D}I(y_i\ne\hat y)=1-\hat\pi_\hat y
$$
Entropy:
$$
H(D)=-\sum_{c=1}^C\hat\pi_c\log_2\hat\pi_c
$$
Gini:
$$
H(D)=\sum_{c=1}^C\hat\pi_c(1-\hat\pi_c)=1-\sum_c\hat\pi_c^2
$$
The Gini coefficient is faster to compute, for it is without logarithm, also it is a better approximation to error rate.

![1555733570103](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1555733570103.png)

### Continuous Input / Regression

The difference between the problem of regression and classification is the difference in $H$. For regression problems, we may have L2 loss function.
$$
H(D)=\sum_{i\in D}(\bar y-y_i)^2,\bar y=\frac{1}{D}\sum_{i\in D}y_i
$$
The closer $y_i$ is to the mean value of the set, the purer the model is.

The result of CART may be huge to look upon.

![1555733649837](C:\Users\a\AppData\Roaming\Typora\typora-user-images\1555733649837.png)

### Terms of Termination

- New splitting gives little gain.

$$
\Delta=cost(D)-(\frac{|D_L|}{|D|}cost(D_L)+\frac{|D_R|}{|D|}cost(D_R))
$$

- Depth of the tree reaches given maximum.
- Left/right subtree is pure enough.
- Left/right subtree contains few enough nodes.

### Pruning

After the construction of the tree, we go from bottom to top to throw away some of the branches of the tree, so that the complexity of the tree may be less. The cost complexity pruning is measured:
$$
CC(T)=Err(T)+\alpha|T|
$$

- $Err$ is the error rate (loss function) of the tree.
- $|T|$ is the number of nodes of the tree.

The coefficient $\alpha$ can be found by cross validation or similar process as other models do.

