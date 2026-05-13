#include <stdio.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {    
    Tensor* a = create_tensor(2, true);
    Tensor* b = create_tensor(3, true);

    Tensor* c = mul(a, b);
    c->grad = 1.0f; // Set the gradient of the output tensor to 1 for backpropagation
    backward(c);

    printf("a.grad: %f\n", a->grad);
    printf("b.grad: %f\n", b->grad);
    printf("c.grad: %f\n", c->grad);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);

    return 0;
}