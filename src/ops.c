#include "ops.h"
#include <stdlib.h>
#include <assert.h>

Tensor* add(Tensor* a, Tensor* b) {
    Tensor* t = create_zeros_tensor(a->shape, a->num_dims, a->requires_grad || b->requires_grad);

    for (u32 i = 0; i < a->size; i++) {
        t->data[i] = a->data[i] + b->data[i];
    }
    t->is_leaf = false;
    
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = ADD;
    }  
    return t;
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* t = create_zeros_tensor(a->shape, a->num_dims, a->requires_grad || b->requires_grad);

    for (u32 i = 0; i < a->size; i++) {
        t->data[i] = a->data[i] * b->data[i];
    }
    t->is_leaf = false;
    
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = MUL;
    }
    return t;
}

Tensor* mat_mul(Tensor* a, Tensor* b) {
    // (m, n) x (n, o) = (m, o)
    u32 m = a->shape[0];
    u32 n = a->shape[1];
    u32 o = b->shape[1];

    assert(n == b->shape[0]);

    u32 shape[2] = {m, o};
    Tensor* t = create_zeros_tensor(shape, 2, a->requires_grad || b->requires_grad);
    for (u32 row = 0; row < m; row++) {
        for (u32 col = 0; col < o; col++) {
            f32 sum = 0.0f;
            for (u32 k = 0; k < n; k++) {
                sum += a->data[row * n + k] * b->data[k * o + col];
            }
            t->data[row * o + col] = sum;
        }
    }
    return t;
}

Tensor* divide(Tensor* a, Tensor* b) {
    Tensor* t = create_zeros_tensor(a->shape, a->num_dims, a->requires_grad || b->requires_grad);

    for (u32 i = 0; i < a->size; i++) {
        t->data[i] = a->data[i] / b->data[i];
    }
    t->is_leaf = false;
    
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = DIV;
    }
    return t;
}