#include "autograd.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

void backward(Tensor* t) {
    if (!t || t->visited) return;
    t->visited = true;

    if (t->grad_op == MUL) {
        mul_backward(t);
    }
    else if (t->grad_op == MAT_MUL) {
        mat_mul_backward(t);
    }
    else if (t->grad_op == DIV) {
        div_backward(t);
    }
    else if (t->grad_op == MAT_ADD) {
        mat_add_backward(t);
    }
    for (u8 i = 0; i < t->num_next_functions; i++) {
        backward(t->next_functions[i]);
    }
}

void mat_add_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];
    
    if (a->requires_grad) {
        for (u32 i = 0; i < a->size; i++) {
            a->grad[i] += t->grad[i];
        }
    }
    if (b->requires_grad) {
        for (u32 i = 0; i < b->size; i++) {
            b->grad[i] += t->grad[i];
        }
    }
}

void mul_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    if (a->requires_grad) {
        a->grad[0] += t->grad[0] * b->data[0];
    }
    if (b->requires_grad) {
        b->grad[0] += t->grad[0] * a->data[0];
    }
}

void div_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    if (a->requires_grad) {
        a->grad[0] += t->grad[0] / b->data[0];
    }
    if (b->requires_grad) {
        b->grad[0] += -t->grad[0] * a->data[0] / (b->data[0] * b->data[0]);
    }
}

void mat_mul_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    u32 m = a->shape[0];
    u32 n = a->shape[1];
    u32 o = b->shape[1];

    // dA = dT x B^t (m, o) x (o, n) = (m, n)
    if (t->shape[1] != b->shape[1]) {
        printf("Shape mismatch: (%u,%u) x (%u,%u)\n", m, o, b->shape[1], b->shape[0]);
        exit(1);
    }
    if (a->requires_grad) {
        for (u32 row = 0; row < m; row++) {
            for (u32 col = 0; col < n; col++) {
                f32 sum = 0.0f;
                for (u32 k = 0; k < o; k++) {
                    sum += t->grad[row * o + k] * b->data[col * o + k];
                } 
                a->grad[row * n + col] += sum;
            }
        }
    }

    // dB = A^t x dT (n, m) x (m, o) = (n, o)
    if (a->shape[0] != t->shape[0]) {
        printf("Shape mismatch: (%u,%u) x (%u,%u)\n", n, m, t->shape[0], t->shape[1]);
        exit(1);
    }
    if (b->requires_grad) {
        for (u32 row = 0; row < n; row++) {
            for (u32 col = 0; col < o; col++) {
                f32 sum = 0.0f;
                for (u32 k = 0; k < m; k++) {
                    sum += a->data[k * n + row] * t->grad[k * o + col];
                }
                b->grad[row * o + col] += sum;
            }
        }
    }
}