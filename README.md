# cytorch
My own implementation of pytorch in C

# Cross Entropy Loss

To improve numerical stability, the softmax (and log-softmax) computation is stabilized by subtracting the maximum logit value before exponentiation.

Let \( x \in \mathbb{R}^n \) be the vector of logits.

The log-softmax is defined as:

\[
\log \text{softmax}(x_i) = \log \left( \frac{e^{x_i}}{\sum_j e^{x_j}} \right)
\]

Expanding the logarithm:

\[
\log \text{softmax}(x_i)
= \log(e^{x_i}) - \log\left(\sum_j e^{x_j}\right)
= x_i - \log\left(\sum_j e^{x_j}\right)
\]

---

## Numerical stability trick

Let:

\[
m = \max(x)
\]

We subtract \( m \) from every logit:

\[
x'_i = x_i - m
\]

This does not change the result because:

\[
\frac{e^{x_i}}{\sum_j e^{x_j}}
=
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
\]

Therefore:

\[
\log \text{softmax}(x_i)
= x_i - m - \log\left(\sum_j e^{x_j - m}\right)
\]

---

## Connection to log-sum-exp

The term:

\[
\log\left(\sum_j e^{x_j}\right)
\]

is known as the **log-sum-exp (LSE)** function, and the stabilized version is:

\[
\log\left(\sum_j e^{x_j}\right)
=
m + \log\left(\sum_j e^{x_j - m}\right)
\]

This prevents numerical overflow when \( x_j \) are large.