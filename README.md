# cytorch
My own implementation of pytorch in C

# Cross Entropy Loss

To improve numerical stability, the softmax (and log-softmax) computation is stabilized by subtracting the maximum logit value before exponentiation.

Let \( x \in \mathbb{R}^n \) be the vector of logits.

The log-softmax is defined as:

$$
\log \text{softmax}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)
$$

Expanding:

$$
\log \text{softmax}(x_i)
= \log(\exp(x_i)) - \log\left(\sum_j \exp(x_j)\right)
= x_i - \log\left(\sum_j \exp(x_j)\right)
$$

---

## Numerical stability trick

Let:

$$
m = \max(x)
$$

We subtract this value from all logits:

$$
x'_i = x_i - m
$$

This transformation does not change the result because:

$$
\frac{\exp(x_i)}{\sum_j \exp(x_j)}
=
\frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}
$$

Therefore, the stabilized log-softmax becomes:

$$
\log \text{softmax}(x_i)
=
x_i - m - \log\left(\sum_j \exp(x_j - m)\right)
$$

---