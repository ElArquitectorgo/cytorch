#include <stdio.h>
#include <assert.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"

bool compare(f32 a, f32 b, f32 tol) {
    return ABS(a - b) < tol;
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

    Tensor* j = mat_add(i, h);

    Tensor* k = mul(j, i);
    k->grad[0] = 1.0f; // Set the gradient of the output tensor to 1 for backpropagation
    backward(k);

    assert(a->data[0] == 2.0);
    assert(b->data[0] == 2.0);
    assert(c->data[0] == 4.0);
    assert(d->data[0] == 2.0);
    assert(e->data[0] == 8.0);
    assert(f->data[0] == 2.0);
    assert(g->data[0] == 16.0);
    assert(h->data[0] == 2.0);
    assert(i->data[0] == 8.0);
    assert(j->data[0] == 10.0);
    assert(k->data[0] == 80.0);

    // GRADS
    assert(a->grad[0] == 0.0);
    assert(b->grad[0] == 0.0);
    assert(c->grad[0] == 36.0);
    assert(d->grad[0] == 0.0);
    assert(e->grad[0] == 18.0);
    assert(f->grad[0] == 0.0);
    assert(g->grad[0] == 9.0);
    assert(h->grad[0] == -64.0);
    assert(i->grad[0] == 18.0);
    assert(j->grad[0] == 8.0);
    assert(k->grad[0] == 1.0);

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

    a = create_tensor((f32[]){1, 2, 3, 1, 3, 1, 1, 1, 1},(u32[]){3, 3}, 2, true);
    b = create_tensor((f32[]){2, 2, 1, 3, 2, 2, 1, 3, 1, 1, 1, 3}, (u32[]){3, 4}, 2, true);
    c = mat_mul(a, b);

    f32 expected[] = {9, 9, 6, 18, 9, 9, 5, 15, 5, 5, 3, 9};
    for (u32 i = 0; i < c->size; i++) {
        assert(c->data[i] == expected[i]);
    }

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);

    a = create_ones_tensor((u32[]){2, 2}, 2, true);
    b = create_ones_tensor((u32[]){3, 4}, 2, true);
    assert(mat_mul(a, b) == NULL);

    free_tensor(a);
    free_tensor(b);

    Tensor* logits = create_tensor((f32[]){2.0, 1.0, 0.1, 0.5, 2.5, 1.0}, (u32[]){2, 3}, 2, true);
    Tensor* labels = create_tensor((f32[]){0, 2}, (u32[]){2, 1}, 2, false);

    Tensor* loss = cross_entropy_loss(logits, labels);
    backward(loss);

    assert(compare(loss->data[0], 1.111693, 1e-5));
    assert(compare(logits->grad[0], -0.170499, 1e-5));
    assert(compare(logits->grad[1], 0.121216, 1e-5));
    assert(compare(logits->grad[2], 0.049283, 1e-5));
    assert(compare(logits->grad[3], 0.049812, 1e-5));
    assert(compare(logits->grad[4], 0.368062, 1e-5));
    assert(compare(logits->grad[5], -0.417874, 1e-5));

    free_tensor(logits);
    free_tensor(labels);
    free_tensor(loss);

    printf("All tests passed!\n");

    return 0;
}