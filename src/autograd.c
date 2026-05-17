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
    else if (t->grad_op == RELU) {
        relu_backward(t);
    }
    else if (t->grad_op == SOFTMAX) {
        softmax_backward(t);
    }
    else if (t->grad_op == CEL) {
        softmax_backward(t);
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

void relu_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];

    if (!a->requires_grad) {
        return;
    }

    for (u32 i = 0; i < a->size; i++) {
        a->grad[i] += (a->data[i] > 0) ? t->grad[i] : 0;
    }
}

void softmax_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];

    if (!a->requires_grad) {
        return;
    }

    f32 sum = 0.0f;
    for (u32 i = 0; i < t->size; i++) {
        sum += t->grad[i] * t->data[i];
    }
    for (u32 i = 0; i < t->size; i++) {
        a->grad[i] += t->data[i] * (t->grad[i] - sum);
    }
}

void cross_entropy_loss_backward(Tensor* t, Tensor* labels) {
    Tensor* logits = t->next_functions[0];

    u32 batch = logits->shape[0];
    u32 classes = logits->shape[1];

    for (u32 b = 0; b < batch; b++) {
        f32 max_val = logits->data[b * classes];
        
        for (u32 c = 1; c < classes; c++) {
            if (logits->data[b * classes + c] > max_val) {
                max_val = logits->data[b * classes + c];
            }
        }
        
        f32 sum = 0.0f;
        for (u32 c = 0; c < classes; c++) {
            sum += expf(logits->data[b * classes + c] - max_val);
        }

        for (u32 c = 0; c < classes; c++) {
            f32 grad = expf(logits->data[b * classes + c] - max_val) / sum;
            u32 target = (u32)labels->data[b];

            if (c == target) {
                grad -= 1.0f;
            }
            grad /= batch;

            logits->grad[b * classes + c] += grad;
        }
    }
}