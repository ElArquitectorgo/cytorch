# cytorch
My own implementation of pytorch in C

# Cross Entropy Loss

To improve numerical stability, the softmax (and log-softmax) computation is stabilized by subtracting the maximum logit value before exponentiation.

Let x be a vector of logits.

The log-softmax is defined as:

log_softmax(x_i) = log(exp(x_i) / sum_j exp(x_j))

Expanding:

log_softmax(x_i)
= log(exp(x_i)) - log(sum_j exp(x_j))
= x_i - log(sum_j exp(x_j))

---

## Numerical stability trick

Let:

m = max(x)

We define:

x'_i = x_i - m

This does not change the result because:

exp(x_i) / sum_j exp(x_j)
= exp(x_i - m) / sum_j exp(x_j - m)

Therefore:

log_softmax(x_i)
= x_i - m - log(sum_j exp(x_j - m))