#include <stdio.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {    
    Tensor* a = create_tensor(2, false);
    Tensor* b = create_tensor(2, false);

    Tensor* c = mul(a, b);
    c->requires_grad = true;
    Tensor* d = create_tensor(2, false);

    Tensor* e = mul(c, d);
    Tensor* f = create_tensor(2, false);

    Tensor* g = mul(e, f);
    Tensor* h = create_tensor(2, true);

    Tensor* i = divide(g, h);

    Tensor* j = add(i, h);

    Tensor* k = mul(j, i);
    k->grad = 1.0f; // Set the gradient of the output tensor to 1 for backpropagation
    backward(k);

    printf("a value: %f, grad: %f\n", a->data, a->grad);
    printf("b value: %f, grad: %f\n", b->data, b->grad);
    printf("c value: %f, grad: %f\n", c->data, c->grad);
    printf("d value: %f, grad: %f\n", d->data, d->grad);
    printf("e value: %f, grad: %f\n", e->data, e->grad);
    printf("f value: %f, grad: %f\n", f->data, f->grad);
    printf("g value: %f, grad: %f\n", g->data, g->grad);
    printf("h value: %f, grad: %f\n", h->data, h->grad);
    printf("i value: %f, grad: %f\n", i->data, i->grad);
    printf("j value: %f, grad: %f\n", j->data, j->grad);
    printf("k value: %f, grad: %f\n", k->data, k->grad);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(d);
    free_tensor(e);
    free_tensor(f);
    free_tensor(g);
    free_tensor(h);
    free_tensor(i);
    free_tensor(j);
    free_tensor(k);
}