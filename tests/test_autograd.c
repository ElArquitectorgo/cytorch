#include <stdio.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"

void assert(const char* name, f32 value, f32 expected) {
    if (value == expected) {
        printf("PASS: %s = %f\n", name, value);
    } else {
        printf("FAIL: %s = %f, expected %f\n", name, value, expected);
    }
}

int main() {    
    Tensor* a = create_1D_tensor(2, false);
    Tensor* b = create_1D_tensor(2, false);

    Tensor* c = mul(a, b);
    c->requires_grad = true;
    Tensor* d = create_1D_tensor(2, false);

    Tensor* e = mul(c, d);
    Tensor* f = create_1D_tensor(2, false);

    Tensor* g = mul(e, f);
    Tensor* h = create_1D_tensor(2, true);

    Tensor* i = divide(g, h);

    Tensor* j = add(i, h);

    Tensor* k = mul(j, i);
    k->grad = 1.0f; // Set the gradient of the output tensor to 1 for backpropagation
    backward(k);

    assert("a.data", a->data[0], 2.0);
    assert("b.data", b->data[0], 2.0);
    assert("c.data", c->data[0], 4.0);
    assert("d.data", d->data[0], 2.0);
    assert("e.data", e->data[0], 8.0);
    assert("f.data", f->data[0], 2.0);
    assert("g.data", g->data[0], 16.0);
    assert("h.data", h->data[0], 2.0);
    assert("i.data", i->data[0], 8.0);
    assert("j.data", j->data[0], 10.0);
    assert("k.data", k->data[0], 80.0);

    // GRADS
    assert("a.grad", a->grad, 0.0);
    assert("b.grad", b->grad, 0.0);
    assert("c.grad", c->grad, 36.0);
    assert("d.grad", d->grad, 0.0);
    assert("e.grad", e->grad, 18.0);
    assert("f.grad", f->grad, 0.0);
    assert("g.grad", g->grad, 9.0);
    assert("h.grad", h->grad, -64.0);
    assert("i.grad", i->grad, 18.0);
    assert("j.grad", j->grad, 8.0);
    assert("k.grad", k->grad, 1.0);

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

    return 0;
}