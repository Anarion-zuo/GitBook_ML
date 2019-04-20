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

A CART is a binary tree, constructed by a recursing process. Instead of 