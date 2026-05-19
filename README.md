# cytorch
My own implementation of pytorch in C

# Cross Entropy Loss

To improve numerical stability, the softmax (and log-softmax) computation is stabilized by subtracting the maximum logit value before exponentiation.

The log-softmax is defined as:

$$
\log \text{softmax}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)
$$

Expanding:

$$
\log \text{softmax}(x_i) = \log(\exp(x_i)) - \log\left(\sum_j \exp(x_j \right) = x_i - \log\left(\sum_j \exp(x_j)\right)
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

Therefore, the stabilized log-softmax becomes:

$$
\log \text{softmax}(x_i) = x_i - m - \log\left(\sum_j \exp(x_j - m)\right)
$$

---

# Optimizer
I'm using the OG formula from Sutskever et al. (2011)

$$
v_{t+1} = \mu v_t - \epsilon \nabla f(\theta_t)
$$
$$
\theta_{t+1} = \theta_t + v_{t+1} 
$$

For some reason, PyTorch uses a different one and doesn't even properly reference the original.

# Test
```python
import torch
import torch.nn.functional as F

# logits shape: (batch, classes)
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 1.0]
], requires_grad=True)

# target class indices
labels = torch.tensor([0, 2])

# cross entropy loss
loss = F.cross_entropy(logits, labels)

print("Loss:")
print(loss.item())

# backward
loss.backward()

print("\nGradients:")
print(logits.grad)
```

```bash
>>> print(loss.item())
1.1116927862167358
>>> print(logits.grad)
tensor([[-0.1705,  0.1212,  0.0493],
        [ 0.0498,  0.3681, -0.4179]])

```

```c
Tensor* logits = create_tensor((f32[]){2.0, 1.0, 0.1, 0.5, 2.5, 1.0}, (u32[]){2, 3}, 2, true);
Tensor* labels = create_tensor((f32[]){0, 2}, (u32[]){2, 1}, 2, false);

Tensor* loss = cross_entropy_loss(logits, labels);
backward(loss);
```

```bash
Loss: 1.111693

Gradients:
logits grad: -0.170499
logits grad: 0.121216
logits grad: 0.049283
logits grad: 0.049812
logits grad: 0.368062
logits grad: -0.417874
```