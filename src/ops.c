#include "ops.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

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

Tensor* mat_mul(Tensor* a, Tensor* b) {
    // (m, n) x (n, o) = (m, o)
    u32 m = a->shape[0];
    u32 n = a->shape[1];
    u32 o = b->shape[1];

    if (n != b->shape[0]) {
        printf("Shape mismatch: (%u,%u) x (%u,%u)\n", m, n, b->shape[0], b->shape[1]);
        return NULL;
    }

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
    
    t->is_leaf = false;
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = MAT_MUL;
    }

    return t;
}

Tensor* mat_add(Tensor* a, Tensor* b) {
    // (m, n) + (m, n) = (m, n)
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1]) {
        printf("Shape mismatch: (%u,%u) + (%u,%u)\n", a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        return NULL;
    }

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
        t->grad_op = MAT_ADD;
    }

    return t;
}

Tensor* ReLU(Tensor* a) {
    Tensor* t = create_zeros_tensor(a->shape, a->num_dims, a->requires_grad);

    for (u32 i = 0; i < a->size; i++) {
        t->data[i] = (a->data[i] > 0) ? a->data[i] : 0;
    }

    t->is_leaf = false;
    t->next_functions = malloc(sizeof(Tensor*));
    t->num_next_functions = 1;
    t->next_functions[0] = a;

    if (t->requires_grad) {
        t->grad_op = RELU;
    }

    return t;

}

Tensor* Softmax(Tensor* a) {
    // e^x_i / sum(e^x_j)
    Tensor* t = create_zeros_tensor(a->shape, a->num_dims, a->requires_grad);

    // To improve numerical stability, we subtract the max value from the input before exponentiating
    f32 max_val = a->data[0];
    for (u32 i = 1; i < a->size; i++) {
        if (a->data[i] > max_val) {
            max_val = a->data[i];
        }
    }

    f32 sum = 0.0f;
    for (u32 i = 0; i < a->size; i++) {
        f32 z = expf(a->data[i] - max_val);
        t->data[i] = z;
        sum += z;
    }

    for (u32 i = 0; i < a->size; i++) {
        t->data[i] /= sum;
    }

    t->is_leaf = false;
    t->next_functions = malloc(sizeof(Tensor*));
    t->num_next_functions = 1;
    t->next_functions[0] = a;

    if (t->requires_grad) {
        t->grad_op = SOFTMAX;
    }

    return t;
}

Tensor* cross_entropy_loss(Tensor* logits, Tensor* labels) {
    Tensor* t = create_zeros_tensor((u32[]){1}, 1, true);
    f32 loss = 0.0f;
    for (u32 i = 0; i < logits->size; i++) {
        loss -= labels->data[i] * logf(logits->data[i] + 1e-8f);
    }
    t->data[0] = loss;
    t->grad[0] = 1.0f;
    t->is_leaf = false;
    t->next_functions = malloc(sizeof(Tensor*));
    t->num_next_functions = 1;
    t->next_functions[0] = logits;
    t->grad_op = None;

    return t;
}